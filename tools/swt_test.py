import pywt
import torch

import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.nn as nn
import torch.nn.functional as F
from SWT import SWTForward,SWTInverse

J = 1
wave = 'db3'
# mode='symmetric'
mode='zero'

# img_1 = plt.imread('00003_1_01660.png')
img_1 = plt.imread('4.jpg')
# img_1 = pywt.data.camera()
# img_2 = pywt.data.ascent()
# img = np.stack([img_1, img_2], 0)
input = torch.tensor(img_1).float()
# input = input[0].reshape(1,1,400,400)
input = input.permute(2, 0, 1)
input = input.unsqueeze(0)
sfm = SWTForward(J, wave, mode)
ifm = SWTInverse(wave, mode)

coeffs = sfm(input)
# ll_0_image = coeffs[0][:,[0,4,8]].reshape(1,3,400,400).float()
# coeffs = sfm(ll_0_image)


recon_R = ifm([coeffs[0][:,0:4]])
recon_G = ifm([coeffs[0][:,4:8]])
recon_B = ifm([coeffs[0][:,8:12]])
recon = torch.cat((recon_R,recon_G,recon_B),dim=1)
ll_image = coeffs[0][:,[0,4,8]].squeeze(0)
plt.imsave('ll_img.png', np.clip(ll_image.permute(1, 2, 0).int().numpy(),0,255).astype(np.uint8))
lh_image =  coeffs[0][:,[1,5,9]].squeeze(0)
lh_image = lh_image + -1*torch.min(lh_image)
plt.imsave('lh_img.png', np.clip(lh_image.permute(1, 2, 0).int().numpy(),0,255).astype(np.uint8))
hl_image =  coeffs[0][:,[2,6,10]].squeeze(0)
hl_image = hl_image + (-1*torch.min(hl_image))
plt.imsave('hl_img.png', np.clip(hl_image.permute(1, 2, 0).int().numpy(),0,255).astype(np.uint8))
hh_image =  coeffs[0][:,[3,7,11]].squeeze(0)
hh_image = hh_image + -1*torch.min(hh_image)
plt.imsave('hh_img.png', np.clip(hh_image.permute(1, 2, 0).int().numpy(),0,255).astype(np.uint8))
print(10 * torch.log10(255**2 / F.mse_loss(input, recon)).item())
plt.subplot(2,3,1)
plt.imshow(input.squeeze(0).permute(1, 2, 0).int())
plt.subplot(2,3,2)
plt.imshow(ll_image.squeeze(0).permute(1, 2, 0).int())
plt.subplot(2,3,3)
plt.imshow(recon.squeeze(0).permute(1, 2, 0).int())
plt.subplot(2,3,4)
plt.imshow(lh_image.squeeze(0).permute(1, 2, 0).int())
plt.subplot(2,3,5)
plt.imshow(hl_image.squeeze(0).permute(1, 2, 0).int())
plt.subplot(2,3,6)
plt.imshow(hh_image.squeeze(0).permute(1, 2, 0).int())
plt.show()