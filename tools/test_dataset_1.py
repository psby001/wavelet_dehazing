import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data
import torchvision.datasets as dates
from torch.autograd import Variable
from torch.nn import functional as F
import shutil
import cv2
from PIL import Image
import tqdm
from einops.einops import rearrange
import math
from torchvision import transforms as transforms1
from torch.optim import lr_scheduler
# import dataset.CD_dataset as dates
from torchmetrics import F1Score
from dz_datasets.loader import PairLoader
import torchvision.transforms as T
import pytorch_ssim
import matplotlib.pyplot as plt
from SWT import SWTForward,SWTInverse
# os.environ['CUDA_LAUNCH_BLOCKING']='1'



def calculate_fft(x):
    fft_im = torch.fft.fft(x.clone())  # bx3xhxw
    fft_amp = fft_im.real**2 + fft_im.imag**2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2(fft_im.imag, fft_im.real)
    return fft_im,fft_amp, fft_pha


def inverse_fft(fft_amp, fft_pha):
    imag = fft_amp * torch.sin(fft_pha)
    real = fft_amp * torch.cos(fft_pha)
    fft_y = torch.complex(real, imag)
    y = torch.fft.ifft(fft_y)
    return y
def main():
    TRANSFROM_SCALES = (400, 400)
    driver ="cuda:0"
    BATCH_SIZE = 8
    # dataset_names = ['RESIDE-6K','RESIDE-IN','RESIDE-OUT',"RSHaze"]
    dataset_names = ['RESIDE-6K']
    J = 1
    wave = 'db1'
    mode = 'zero'
    SsimLoss = pytorch_ssim.SSIM().to(driver)
    # sfm = SWTForward(J, wave, mode).to(driver)
    # ifm = SWTInverse(wave, mode).to(driver)
    from wavelet import iwt_m,wt_m
    sfm = wt_m().to(driver)
    ifm = iwt_m().to(driver)
    for dataset_name in dataset_names:
        for phase in ['train', 'test']:
            torch.cuda.empty_cache()
            DATA_PATH = '/mnt/d/Train Data/dz_data/' + dataset_name + '/'
            test_data = PairLoader(DATA_PATH, phase, 'test',
                                   TRANSFROM_SCALES)
            val_loader = Data.DataLoader(test_data, batch_size=32,
                                         shuffle=False, num_workers=4)
            print(len(test_data), len(val_loader))
            ORG_PSNR = []
            PSNR_S1 = []
            ORG_SSMI = []
            SSMI_S1 = []
            for batch_idx, batch in tqdm.tqdm(enumerate(val_loader)):
                img_idx, label_idx = batch["source"].to(driver), batch["target"].to(driver)
                fft_im_img, fft_amp_img, fft_pha_img = calculate_fft(img_idx)
                fft_im_label, fft_amp_label, fft_pha_label = calculate_fft(label_idx)
                recon = inverse_fft(fft_amp_label, fft_pha_img).real
                # coeffs_label = sfm(label_idx)
                # ll_label = coeffs_label[:, [0, 4, 8]]
                # detail_label = coeffs_label[:, [1, 2, 3, 5, 6, 7, 9, 10, 11]]
                # coeffs_img = sfm(img_idx)
                # ll_img = coeffs_img[:, [0, 4, 8]]
                # detail_img = coeffs_img[:, [1, 2, 3, 5, 6, 7, 9, 10, 11]]
                # recon_R = ifm(torch.cat((ll_label[:, [0]], detail_img[:, 0:3]), dim=1))
                # recon_G = ifm(torch.cat((ll_label[:, [1]], detail_img[:, 3:6]), dim=1))
                # recon_B = ifm(torch.cat((ll_label[:, [2]], detail_img[:, 6:9]), dim=1))
                # recon = torch.cat((recon_R, recon_G, recon_B), dim=1)


                ORG_PSNR.append(10 * torch.log10(1 / F.mse_loss(img_idx, label_idx)).item())
                ORG_SSMI.append(SsimLoss(img_idx, label_idx).item())
                PSNR_S1.append(10 * torch.log10(1 / F.mse_loss(recon, label_idx)).item())
                SSMI_S1.append(SsimLoss(recon, label_idx).item())
            print("ORG_PSNR:",np.mean(ORG_PSNR),"PSNR_S1:",np.mean(PSNR_S1))
            print("ORG_SSMI:", np.mean(ORG_SSMI), "SSMI_S1:", np.mean(SSMI_S1))





if __name__ == '__main__':
    main()