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

# os.environ['CUDA_LAUNCH_BLOCKING']='1'




def main():
    TRANSFROM_SCALES = (400, 400)
    driver ="cuda:0"
    BATCH_SIZE = 8
    # dataset_names = ['RESIDE-6K','RESIDE-IN','RESIDE-OUT',"RSHaze"]
    dataset_names = ['RESIDE-OUT']
    for dataset_name in dataset_names:
        for phase in ['train', 'test']:
            torch.cuda.empty_cache()
            DATA_PATH = '/mnt/d/Train Data/dz_data/' + dataset_name + '/'
            test_data = PairLoader(DATA_PATH, phase, 'test',
                                   TRANSFROM_SCALES)
            val_loader = Data.DataLoader(test_data, batch_size=32,
                                         shuffle=False, num_workers=4)
            print(len(test_data), len(val_loader))
            list_ = []
            for batch_idx, batch in tqdm.tqdm(enumerate(val_loader)):
                # if batch_idx >= 100:
                #     break
                # img_idx, label_idx ,name= batch["source"].astype(int), batch["target"].astype(int),batch["filename"]
                # loss = np.abs(label_idx - img_idx)
                # file_name = name.rsplit(".",1)[0]
                # img = cv2.cvtColor(img_idx.transpose(1, 2, 0).astype(np.uint8), cv2.COLOR_RGB2BGR)
                # heatmap_R = cv2.applyColorMap(loss[2].astype(np.uint8), cv2.COLORMAP_JET)
                # heatmap_G = cv2.applyColorMap(loss[1].astype(np.uint8), cv2.COLORMAP_JET)
                # heatmap_B = cv2.applyColorMap(loss[1].astype(np.uint8), cv2.COLORMAP_JET)
                # cv2.imwrite(file_name + '.png', img)
                # cv2.imwrite(file_name + '_R.png', heatmap_R)
                # cv2.imwrite(file_name + '_G.png', heatmap_G)
                # cv2.imwrite(file_name + '_B.png', heatmap_B)

                img_idx, label_idx = batch["source"].to(driver), batch["target"].to(driver)
                loss = label_idx - img_idx
                loss = loss*loss
                # list_.extend(torch.sum(loss,dim = (1,2,3)).cpu().numpy().tolist())
                loss = torch.mean(loss,dim = (1,2,3))
                list_.extend((10 * torch.log10(1 / loss)).cpu().numpy().tolist())

            # plt.hist(list_, bins=range(0,31,2), alpha=0.3, label=dataset_name)
            bins = range(0, 31, 2)
            # bins=range(0,200000,5000)
            counts, bin = np.histogram(list_, bins=bins)
            counts = counts/len(test_data)
            plt.stairs(counts, bins,label=phase,fill = True,alpha=0.3)
            # plt.xticks(range(0,31,2))
            # plt.savefig(dataset_name + "_see" + ".png")
            # plt.clf()
        # plt.xticks(bins)
        plt.xlabel("PSNR")
        plt.ylabel("Number of Picture")
        plt.legend()
        plt.savefig(dataset_name + "_psnr.png")
        plt.clf()





if __name__ == '__main__':
    main()