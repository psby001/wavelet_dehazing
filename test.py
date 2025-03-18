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
from fvcore.nn import FlopCountAnalysis
from collections import OrderedDict

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




def main(args):
    TRANSFROM_SCALES = (args.TRANSFROM_SCALES, args.TRANSFROM_SCALES)
    dataset_name = args.DATA_PATH.split('/')[-1]
    ab_test_dir = check_dir(os.path.join(args.OUTPUT_PATH, dataset_name))

    model_path = args.weight_path.rsplit("/",1)[0]
    if os.path.exists(os.path.join(model_path, "net.py")):
        print("Loading weights from {}".format(model_path))
        import importlib
        import sys
        spec = importlib.util.spec_from_file_location("net", os.path.join(model_path, "net.py"))
        net = importlib.util.module_from_spec(spec)
        sys.modules["net"] = net
        spec.loader.exec_module(net)
        model = net.UNet_wavelet()
        # state_dict = torch.load(args.weight_path)["state_dict"]
        # new_state_dict= {}
        # for k, v in state_dict.items():
        #     if "strip_att.fushion" in k:
        #         continue
        #     name = k
        #     new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。
        # model.load_state_dict(new_state_dict)
        model.load_state_dict(torch.load(args.weight_path)["state_dict"])
    else:
        model = torch.load(args.weight_path)
    # model = UNet()
    # model = torch.load(args.weight_path)
    # model.load_state_dict(torch.load(weight_path)["state_dict"])
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if args.half:
        model.half()
    model.to(device)
    model.eval()
    # input = torch.randn(1, 3, args.TRANSFROM_SCALES,args.TRANSFROM_SCALES).to(device)
    # if args.half:
    #     input = input.half()
    # flops = FlopCountAnalysis(model, input)
    # print(flops.total()/1000/1000/1000)
    model_name = model.__class__.__name__
    print(model_name)
    test_data = PairLoader(args.DATA_PATH, 'test', 'valid',
                           TRANSFROM_SCALES)
    test_loader = Data.DataLoader(test_data, batch_size=1,
                                 shuffle=False, num_workers=4, pin_memory=True)

    Cw_SsimLoss = CW_SSIM(imgSize=TRANSFROM_SCALES, channels=3, level=4, ori=8)
    Cw_SsimLoss_L = CW_SSIM(imgSize=(args.TRANSFROM_SCALES//2, args.TRANSFROM_SCALES//2), channels=3, level=4, ori=8)
    Cw_SsimLoss_D = CW_SSIM(imgSize=(args.TRANSFROM_SCALES//2, args.TRANSFROM_SCALES//2), channels=9, level=4, ori=8)
    SsimLoss = pytorch_ssim.SSIM()
    # if args.half:
    #     Cw_SsimLoss.half()
    #     Cw_SsimLoss_L.half()
    #     Cw_SsimLoss_D.half()
    #     SsimLoss.half()
    Cw_SsimLoss.to(device)
    Cw_SsimLoss_L.to(device)
    Cw_SsimLoss_D.to(device)
    SsimLoss.to(device)
    print("test_loader", len(test_loader))

    total_loss = 0.
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
    transform = T.ToPILImage()

    sfm = wt_m()
    # sfm = SWTForward()
    ifm = iwt_m()
    # ifm = SWTInverse()
    sfm.to(device)
    ifm.to(device)
    if args.half:
        # sfm.half()
        ifm.half()
    for batch_idx, batch in tqdm.tqdm(enumerate(test_loader)):
        with torch.no_grad():
            img_idx, label_idx ,names= batch["source"], batch["target"],batch["filename"]
            # img = Variable(img_idx.to(device))
            img = img_idx
            # label = Variable(label_idx.to(device))
            label = label_idx
            if args.half:
                img = img.half()
                # label = label.half()
            img = img.to(device)
            label= label.to(device)
            coeffs = sfm(label)
            ll_label = coeffs[:, [0, 4, 8]]
            detail_label = coeffs[:, [1, 2, 3, 5, 6, 7, 9, 10, 11]]
            start_time = time.time()
            (output_map,enc1, enc2, enc3, enc4) = model(img)
            end_time = time.time()
            total_time.append(end_time - start_time)
            out_ll,out_detail = enc1

            # ll = ll[0]
            # recon_R = ifm(torch.cat((ll[:, [0]], detail[:, 0:3]), dim=1))
            # recon_G = ifm(torch.cat((ll[:, [1]], detail[:, 3:6]), dim=1))
            # recon_B = ifm(torch.cat((ll[:, [2]], detail[:, 6:9]), dim=1))
            # output_map = torch.cat((recon_R, recon_G, recon_B), dim=1)
            if args.half:
                out_ll = out_ll.to(torch.float32)
                out_detail = out_detail.to(torch.float32)
                output_map = output_map.to(torch.float32)

            total_psnr += (10 * torch.log10(1 / F.mse_loss(output_map, label)).item())
            total_ll_psnr += (10 * torch.log10(1 / F.mse_loss(out_ll, ll_label)).item())
            total_detail_psnr += (10 * torch.log10(1 / F.mse_loss(out_detail, detail_label)).item())
            total_ssim += (SsimLoss(output_map, label).item())
            total_ll_ssim += (SsimLoss(out_ll, ll_label).item())
            total_detail_ssim += (SsimLoss(out_detail, detail_label).item())
            total_cw_ssim += (Cw_SsimLoss.cw_ssim(output_map, label).item())
            total_ll_cw_ssim += (Cw_SsimLoss_L.cw_ssim(out_ll, ll_label).item())
            total_detail_cw_ssim += (Cw_SsimLoss_D.cw_ssim(out_detail, detail_label).item())
            memory_use.append((torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024)+(torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024)+(torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))
            # for out_img,lls,details,hz_img,gt_img,name in zip(output_map,out_ll,out_detail,img,label,names):
            #     if not os.path.exists(ab_test_dir):
            #         os.makedirs(ab_test_dir)
            #     out_image = transform(out_img)
            #     out_image = np.asarray(out_image)
            #     out_image = Image.fromarray(out_image)
            #     out_image.save(os.path.join(ab_test_dir, name))
            #     hz_image = transform(hz_img)
            #     hz_image = np.asarray(hz_image)
            #     hz_image = Image.fromarray(hz_image)
            #     hz_image.save(os.path.join(ab_test_dir, "hz_" + name))
            #     gz_image = transform(gt_img)
            #     gz_image = np.asarray(gz_image)
            #     gz_image = Image.fromarray(gz_image)
            #     gz_image.save(os.path.join(ab_test_dir, "gz_" + name))
                # ll_image = transform(lls)
                # ll_image = np.asarray(ll_image)
                # ll_image = Image.fromarray(ll_image)
                # ll_image.save(os.path.join(ab_test_dir, "ll_"+name))
                # lh = details[[0,3,6]]
                # hl = details[[1, 4, 7]]
                # hh = details[[2, 5, 8]]
                # lh_image = transform(lh)
                # lh_image = np.asarray(lh_image)
                # lh_image = Image.fromarray(lh_image)
                # lh_image.save(os.path.join(ab_test_dir, "lh_" + name))
                # hl_image = transform(hl)
                # hl_image = np.asarray(hl_image)
                # hl_image = Image.fromarray(hl_image)
                # hl_image.save(os.path.join(ab_test_dir, "hl_" + name))
                # hh_image = transform(hh)
                # hh_image = np.asarray(hh_image)
                # hh_image = Image.fromarray(hh_image)
                # hh_image.save(os.path.join(ab_test_dir, "hh_" + name))



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
    args.add_argument('--weight_path', type=str, default="output/RESIDE-6K_UNet_wavelet_56/model_best.pth",
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