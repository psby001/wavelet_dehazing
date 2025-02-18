import argparse
import numpy as np
import os
import torch
from dz_datasets.loader import PairLoader
import torch.utils.data as Data
from torch.nn import functional as F
from PIL import Image
import tqdm
from SWT import SWTForward,SWTInverse
from wavelet import wt_m,iwt_m
import math
import torchvision.transforms as T
import pytorch_ssim
import matplotlib.pyplot as plt
import time
from IQA_pytorch import CW_SSIM

import tensorrt as trt
from collections import OrderedDict,namedtuple
def check_dir(path,n=0):
    if (not os.path.exists(path)) and n==0:
        # os.makedirs(path)
        return path
    elif not os.path.exists(path+"{:0>2d}".format(n)):
        # os.makedirs(path+"{:0>2d}".format(n))
        return path+"{:0>2d}".format(n)
    else:
        n+=1
        return check_dir(path,n)

def getBindings(model,context,device):
    bindings = OrderedDict()
    Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
    for i in range(model.num_io_tensors):
        name = model.get_tensor_name(i)
        shape = trt.volume(model.get_tensor_shape(name))
        dtype = trt.nptype(model.get_tensor_dtype(name))
        data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
        bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
        print(name,dtype,shape)
    # for tensorrt 8
    # for index in range(model.num_bindings):
    #     name = model.get_binding_name(index)
    #     dtype = trt.nptype(model.get_binding_dtype(index))
    #     shape = tuple(context.get_binding_shape(index))
    #     data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
    #     bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
    #     # print(name,dtype,shape)
    return bindings

def main(args):
    TRANSFROM_SCALES = (args.TRANSFROM_SCALES, args.TRANSFROM_SCALES)
    dataset_name = args.DATA_PATH.split('/')[-1]
    ab_test_dir = check_dir(os.path.join(args.OUTPUT_PATH, dataset_name))
    device = torch.device("cuda:0")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
        # trt_engine = trt.utils.load_engine(G_LOGGER, engine_path)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    success = parser.parse_from_file(args.weight_path)
    config = builder.create_builder_config()
    # config.max_workspace_size = 1 << 30  # 1GB
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    # engine = builder.build_engine(network, config)
    engine = builder.build_engine_with_config(network, config)
    context = engine.create_execution_context()
    bindings = getBindings(engine, context,device)
    binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())


    test_data = PairLoader(args.DATA_PATH, 'test', 'valid',
                           TRANSFROM_SCALES)
    test_loader = Data.DataLoader(test_data, batch_size=1,
                                 shuffle=False, num_workers=4, pin_memory=True)

    Cw_SsimLoss = CW_SSIM(imgSize=TRANSFROM_SCALES, channels=3, level=4, ori=8).to(device)
    Cw_SsimLoss_L = CW_SSIM(imgSize=(args.TRANSFROM_SCALES//2, args.TRANSFROM_SCALES//2), channels=3, level=4, ori=8).to(device)
    Cw_SsimLoss_D = CW_SSIM(imgSize=(args.TRANSFROM_SCALES//2, args.TRANSFROM_SCALES//2), channels=9, level=4, ori=8).to(device)
    SsimLoss = pytorch_ssim.SSIM().to(device)
    print("test_loader", len(test_loader))
    transform = T.ToPILImage()
    total_psnr = 0.
    total_ll_psnr = 0.
    total_detail_psnr = 0.
    total_ssim = 0.
    total_ll_ssim = 0.
    total_detail_ssim = 0.
    total_cw_ssim = 0.
    total_ll_cw_ssim = 0.
    total_detail_cw_ssim = 0.
    total_time = []
    memory_use = []

    sfm = wt_m()
    # sfm = SWTForward()
    ifm = iwt_m()
    # ifm = SWTInverse()
    sfm.to(device)
    ifm.to(device)
    model_name = args.weight_path
    for i in tqdm.tqdm(range(10)):
        input_tensor = torch.randn(1, 3, 256, 256).to(device)
        binding_addrs['input'] = int(input_tensor.data_ptr())
        context.execute_v2(list(binding_addrs.values()))
    keys = list(bindings.keys())
    for batch_idx, batch in tqdm.tqdm(enumerate(test_loader)):
        with torch.no_grad():
            img_idx, label_idx ,names= batch["source"], batch["target"],batch["filename"]
            # img = Variable(img_idx.to(device))
            img = img_idx.to(device)
            label = label_idx.to(device)
            if args.half:
                img = img.half()
            binding_addrs['input'] = int(img.data_ptr())
            start_time = time.time()
            context.execute_v2(list(binding_addrs.values()))
            end_time = time.time()
            total_time.append(end_time - start_time)
            out_ll = bindings[keys[2]].data.to(torch.float).view(1, 3, 128, 128)
            out_detail = bindings[keys[3]].data.to(torch.float).view(1, 9, 128, 128)
            output_map = bindings[keys[1]].data.to(torch.float).view(1, 3, 256, 256)
            coeffs = sfm(label)
            ll_label = coeffs[:, [0, 4, 8]]
            detail_label = coeffs[:, [1, 2, 3, 5, 6, 7, 9, 10, 11]]
            # ll = ll[0]
            # recon_R = ifm(torch.cat((ll[:, [0]], detail[:, 0:3]), dim=1))
            # recon_G = ifm(torch.cat((ll[:, [1]], detail[:, 3:6]), dim=1))
            # recon_B = ifm(torch.cat((ll[:, [2]], detail[:, 6:9]), dim=1))
            # output_map = torch.cat((recon_R, recon_G, recon_B), dim=1)
            total_psnr += (10 * torch.log10(1 / F.mse_loss(output_map, label)).item())
            total_ll_psnr += (10 * torch.log10(1 / F.mse_loss(out_ll, ll_label)).item())
            total_detail_psnr += (10 * torch.log10(1 / F.mse_loss(out_detail, detail_label)).item())
            total_ssim += (SsimLoss(output_map, label).item())
            total_ll_ssim += (SsimLoss(out_ll, ll_label).item())
            total_detail_ssim += (SsimLoss(out_detail, detail_label).item())
            total_cw_ssim += (Cw_SsimLoss.cw_ssim(output_map, label).item())
            total_ll_cw_ssim += (Cw_SsimLoss_L.cw_ssim(out_ll, ll_label).item())
            total_detail_cw_ssim += (Cw_SsimLoss_D.cw_ssim(out_detail, detail_label).item())
            for out_img,lls,details,hz_img,gt_img,name in zip(output_map,out_ll,out_detail,img,label,names):
                if not os.path.exists(ab_test_dir):
                    os.makedirs(ab_test_dir)
                out_image = transform(out_img)
                out_image = np.asarray(out_image)
                out_image = Image.fromarray(out_image)
                out_image.save(os.path.join(ab_test_dir, name))
            #     hz_image = transform(hz_img)
            #     hz_image = np.asarray(hz_image)
            #     hz_image = Image.fromarray(hz_image)
            #     hz_image.save(os.path.join(ab_test_dir, "hz_" + name))
            #     gz_image = transform(gt_img)
            #     gz_image = np.asarray(gz_image)
            #     gz_image = Image.fromarray(gz_image)
            #     gz_image.save(os.path.join(ab_test_dir, "gz_" + name))
            #     # ll_image = transform(lls)
            #     # ll_image = np.asarray(ll_image)
            #     # ll_image = Image.fromarray(ll_image)
            #     # ll_image.save(os.path.join(ab_test_dir, "ll_"+name))
            #     # lh = details[[0,3,6]]
            #     # hl = details[[1, 4, 7]]
            #     # hh = details[[2, 5, 8]]
            #     # lh_image = transform(lh)
            #     # lh_image = np.asarray(lh_image)
            #     # lh_image = Image.fromarray(lh_image)
            #     # lh_image.save(os.path.join(ab_test_dir, "lh_" + name))
            #     # hl_image = transform(hl)
            #     # hl_image = np.asarray(hl_image)
            #     # hl_image = Image.fromarray(hl_image)
            #     # hl_image.save(os.path.join(ab_test_dir, "hl_" + name))
            #     # hh_image = transform(hh)
            #     # hh_image = np.asarray(hh_image)
            #     # hh_image = Image.fromarray(hh_image)
            #     # hh_image.save(os.path.join(ab_test_dir, "hh_" + name))




    print("############################")
    print("SSMI ", total_ssim/len(test_loader),"CW_SSMI ", total_cw_ssim/len(test_loader) ,"PSNR ",total_psnr/len(test_loader))
    print("LL_SSMI ", total_ll_ssim / len(test_loader),"LL_CW_SSMI ", total_ll_cw_ssim / len(test_loader), "LL_PSNR ",
          total_ll_psnr / len(test_loader))
    print("Detail_SSMI ", total_detail_ssim / len(test_loader),"Detail_CW_SSMI ", total_detail_cw_ssim / len(test_loader), "Detail_PSNR ",
          total_detail_psnr / len(test_loader))
    print("avg inference time:",np.mean(total_time))
    print("avg GPU Memory:", np.mean(memory_use))

def opt_args():
    args = argparse.ArgumentParser()
    args.add_argument('--DATA_PATH', type=str, default="/mnt/d/Train Data/dz_data/RESIDE-6K",
                      help='Path to Dataset')
    args.add_argument('--weight_path', type=str, default="output_UNet_wavelet.onnx",
                      help='Path to model weight')
    args.add_argument('--OUTPUT_PATH', type=str, default="./result",
                      help='Output Path')
    args.add_argument('--TRANSFROM_SCALES', type=int, default=256,
                      help='train img size')
    args.add_argument('--half', type=bool, default=False,
                      help='use float16')
    return args.parse_args()

if __name__ == '__main__':
    opt = opt_args()
    main(opt)