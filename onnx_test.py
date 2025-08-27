"""
使用方式：
    python onnx_test.py --weight_path <模型權重路徑> --DATA_PATH <資料集路徑> \
                        [--OUTPUT_PATH ./result] [--TRANSFROM_SCALES 256] [--half True]

參數說明：
    --weight_path      ONNX 權重檔路徑 (必填)
    --DATA_PATH        測試資料集路徑 (必填)
    --OUTPUT_PATH      輸出結果目錄，預設 ./result
    --TRANSFROM_SCALES 輸入影像尺寸，預設 256
    --half             是否使用半精度推論，預設 False
"""

import argparse
import numpy as np
import os
import torch
import onnxruntime as ort
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ort.InferenceSession(args.weight_path, providers=[
                                               "CUDAExecutionProvider",
                                               "CPUExecutionProvider"       # 使用CPU推理
                                           ])

    test_data = PairLoader(args.DATA_PATH, 'test', 'valid',
                           TRANSFROM_SCALES)
    test_loader = Data.DataLoader(test_data, batch_size=1,
                                 shuffle=False, num_workers=4, pin_memory=True)

    Cw_SsimLoss = CW_SSIM(imgSize=TRANSFROM_SCALES, channels=3, level=4, ori=8).to(device)
    Cw_SsimLoss_L = CW_SSIM(imgSize=(args.TRANSFROM_SCALES//2, args.TRANSFROM_SCALES//2), channels=3, level=4, ori=8).to(device)
    Cw_SsimLoss_D = CW_SSIM(imgSize=(args.TRANSFROM_SCALES//2, args.TRANSFROM_SCALES//2), channels=9, level=4, ori=8).to(device)
    SsimLoss = pytorch_ssim.SSIM().to(device)
    if args.half:
        Cw_SsimLoss.half()
        Cw_SsimLoss_L.half()
        Cw_SsimLoss_D.half()
        SsimLoss.half()
    print("test_loader", len(test_loader))

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
    if args.half:
        sfm.half()
        ifm.half()
    model_name = args.weight_path
    for i in tqdm.tqdm(range(10)):
        input_tensor = np.random.rand(1,3,args.TRANSFROM_SCALES,args.TRANSFROM_SCALES).astype(np.float32)
        if args.half:
            input_tensor =input_tensor.astype(np.float16)
        ort_inputs = {model.get_inputs()[0].name: input_tensor}
        if "wavelet" in model_name:
            (output_map, out_ll, out_detail, dec2_out, dec3_out, enc4) = model.run(None, ort_inputs)
        else:
            (output_map, out_detail, dec2_out, dec3_out, enc4) = model.run(None, ort_inputs)
    for batch_idx, batch in tqdm.tqdm(enumerate(test_loader)):
        with torch.no_grad():
            img_idx, label_idx ,names= batch["source"], batch["target"],batch["filename"]
            # img = Variable(img_idx.to(device))
            img = img_idx.detach().cpu().numpy()
            label_idx = label_idx.to(device)
            if args.half:
                img = img.astype(np.float16)
                label_idx = label_idx.half()

            if "wavelet" in model_name:
                ort_inputs = {model.get_inputs()[0].name: img}
                start_time = time.time()
                (output_map, out_ll, out_detail, dec2_out, dec3_out, enc4) = model.run(None, ort_inputs)
                end_time = time.time()
                total_time.append(end_time - start_time)
                coeffs = sfm(label_idx)
                ll_label = coeffs[:, [0, 4, 8]]
                detail_label = coeffs[:, [1, 2, 3, 5, 6, 7, 9, 10, 11]]
                # ll = ll[0]
                # recon_R = ifm(torch.cat((ll[:, [0]], detail[:, 0:3]), dim=1))
                # recon_G = ifm(torch.cat((ll[:, [1]], detail[:, 3:6]), dim=1))
                # recon_B = ifm(torch.cat((ll[:, [2]], detail[:, 6:9]), dim=1))
                # output_map = torch.cat((recon_R, recon_G, recon_B), dim=1)
                output_map = torch.from_numpy(output_map).to(device)
                out_ll = torch.from_numpy(out_ll).to(device)
                out_detail = torch.from_numpy(out_detail).to(device)
                total_psnr += (10 * torch.log10(1 / F.mse_loss(output_map, label_idx)).item())
                total_ll_psnr += (10 * torch.log10(1 / F.mse_loss(out_ll, ll_label)).item())
                total_detail_psnr += (10 * torch.log10(1 / F.mse_loss(out_detail, detail_label)).item())
                total_ssim += (SsimLoss(output_map, label_idx).item())
                total_ll_ssim += (SsimLoss(out_ll, ll_label).item())
                total_detail_ssim += (SsimLoss(out_detail, detail_label).item())
                total_cw_ssim += (Cw_SsimLoss.cw_ssim(output_map, label_idx).item())
                total_ll_cw_ssim += (Cw_SsimLoss_L.cw_ssim(out_ll, ll_label).item())
                total_detail_cw_ssim += (Cw_SsimLoss_D.cw_ssim(out_detail, detail_label).item())

            else:
                ort_inputs = {model.get_inputs()[0].name: img}
                start_time = time.time()
                (output_map, enc1, enc2, enc3, enc4) = model.run(None, ort_inputs)
                end_time = time.time()
                total_time.append(end_time - start_time)
                output_map = torch.from_numpy(output_map).to(device)
                total_psnr += (10 * torch.log10(1 / F.mse_loss(output_map, label_idx)).item())
                total_ssim += (SsimLoss(output_map, label_idx).item())
                total_cw_ssim += (Cw_SsimLoss.cw_ssim(output_map, label_idx).item())




    print("############################")
    print("SSMI ", total_ssim/len(test_loader),"CW_SSMI ", total_cw_ssim/len(test_loader) ,"PSNR ",total_psnr/len(test_loader))
    if "wavelet" in model_name:
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
