import argparse
import numpy as np
import os
import sys
import torch

import net
from dz_datasets.loader import PairLoader
import torch.utils.data as Data
from torch.nn import functional as F
from PIL import Image
import tqdm
from SWT import SWTForward,SWTInverse
from wavelet import wt_m,iwt_m
import math
from torchmetrics import F1Score
import torchvision.transforms as T
import pytorch_ssim
import matplotlib.pyplot as plt
import time
from IQA_pytorch import CW_SSIM
from fvcore.nn import FlopCountAnalysis


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


    # model = UNet()
    model_path = args.weight_path.rsplit("/")[-1]
    if os.path.exists(os.path.join(model_path,"net.py")):
        import importlib
        spec = importlib.util.spec_from_file_location("net", os.path.join(model_path,"net.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules["net"] = module
        spec.loader.exec_module(module)
        if "wavelet" in args.weight_path:
            model = net.UNet_wavelet()
            model.load_state_dict(torch.load(args.weight_path))
        else:
            model = net.UNet()
            model.load_state_dict(torch.load(args.weight_path))
    else:
        model = torch.load(args.weight_path)
    # model = UNet_wavelet()
    # model.load_state_dict(torch.load(args.weight_path))
    # model.load_state_dict(torch.load(weight_path)["state_dict"])
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()
    # input = torch.randn(1, 3, args.TRANSFROM_SCALES,args.TRANSFROM_SCALES).to(device)
    # flops = FlopCountAnalysis(model, input)
    # print(flops.total()/1024/1024/1024)
    model_name = model.__class__.__name__
    print(model_name)
    test_data = PairLoader(args.DATA_PATH, 'test', 'valid',
                           TRANSFROM_SCALES)
    test_loader = Data.DataLoader(test_data, batch_size=1,
                                 shuffle=False, num_workers=4, pin_memory=True)

    print("test_loader", len(test_loader))
    total_time_stage_0 = [0]
    total_time_stage_1 = [0]
    total_time_stage_2 = [0]
    total_time_stage_3 = [0]
    total_time_stage_4 = [0]
    memory_use = []

    sfm = wt_m()
    # sfm = SWTForward()
    ifm = iwt_m()
    # ifm = SWTInverse()
    sfm.to(device)
    ifm.to(device)
    for batch_idx, batch in tqdm.tqdm(enumerate(test_loader)):
        with torch.no_grad():
            img_idx, label_idx ,names= batch["source"], batch["target"],batch["filename"]
            # img = Variable(img_idx.to(device))
            img = img_idx.to(device)
            # label = Variable(label_idx.to(device))
            label = label_idx.to(device)
            if "wavelet" in model_name:
                start_time0 = time.time()
                x = model.forward_stage_0(img)
                x = x.contiguous()
                end_time0 = time.time()
                total_time_stage_0.append(end_time0-start_time0)
                start_time1 = time.time()
                (enc1, enc2, enc3, enc4) = model.forward_stage_1(x)
                end_time1 = time.time()
                total_time_stage_1.append(end_time1-start_time1)
                start_time2 = time.time()
                ll = model.forward_stage_2(enc1, enc2, enc3, enc4)
                end_time2 = time.time()
                total_time_stage_2.append(end_time2-start_time2)
                start_time3 = time.time()
                detail = model.forward_stage_3(enc1, enc2, enc3, enc4)
                end_time3 = time.time()
                total_time_stage_3.append(end_time3-start_time3)
                start_time4 = time.time()
                output_img = model.forward_stage_4(ll,detail)
                end_time4 = time.time()
                total_time_stage_4.append(end_time4-start_time4)
                memory_use.append((torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024)+(torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024)+(torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))

            else:
                start_time1 = time.time()
                (enc1, enc2, enc3, enc4) = model.forward_stage_1(img)
                end_time1 = time.time()
                total_time_stage_1.append(end_time1-start_time1)
                start_time2 = time.time()
                output_map = model.forward_stage_2(enc1, enc2, enc3, enc4)
                end_time2 = time.time()
                total_time_stage_2.append(end_time2-start_time2)
                memory_use.append((torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024) + (
                        torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024) + (
                                          torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))



    print("############################")
    print("avg total inference time:",np.mean(total_time_stage_0)+np.mean(total_time_stage_1)+np.mean(total_time_stage_2)+np.mean(total_time_stage_3)+np.mean(total_time_stage_4))
    print("avg stage0 inference time:",np.mean(total_time_stage_0))
    print("avg stage1 inference time:",np.mean(total_time_stage_1))
    print("avg stage2 inference time:",np.mean(total_time_stage_2))
    print("avg stage3 inference time:",np.mean(total_time_stage_3))
    print("avg stage4 inference time:",np.mean(total_time_stage_4))
    print("avg GPU Memory:", np.mean(memory_use))
    # plt.subplot(2, 3, 1)
    # plt.plot(range(len(total_time_stage_0)),total_time_stage_0)
    # plt.title("Population Growth")
    # plt.subplot(2, 3, 2)
    # plt.plot(range(len(total_time_stage_1)),total_time_stage_1)
    # plt.subplot(2, 3, 3)
    # plt.plot(range(len(total_time_stage_2)),total_time_stage_2)
    # plt.subplot(2, 3, 4)
    # plt.plot(range(len(total_time_stage_3)),total_time_stage_3)
    # plt.subplot(2, 3, 5)
    # plt.plot(range(len(total_time_stage_4)), total_time_stage_4)
    # plt.show()

def opt_args():
    args = argparse.ArgumentParser()
    args.add_argument('--DATA_PATH', type=str, default="/mnt/d/Train Data/dz_data/RESIDE-6K",
                      help='Path to Dataset')
    args.add_argument('--weight_path', type=str, default="output/RESIDE-6K_UNet_wavelet_20/model_best.pth",
                      help='Path to model weight')
    args.add_argument('--OUTPUT_PATH', type=str, default="./result",
                      help='Output Path')
    args.add_argument('--TRANSFROM_SCALES', type=int, default=256,
                      help='train img size')
    return args.parse_args()

if __name__ == '__main__':
    opt = opt_args()
    main(opt)