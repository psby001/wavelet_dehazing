import torch
import cv2
from future.moves import sys
from torch.nn import functional as F
import matplotlib.pyplot as plt
# from pytorch_wavelets import DWTForward, DWTInverse # (or import DWT, IDWT)
from SWT import DWTForward, DWTInverse
from wavelet import wt_m,iwt_m
# xfm = DWTForward(J=1, mode='zero', wave='db3')  # Accepts all wave types available to PyWavelets
# ifm = DWTInverse(mode='zero', wave='db3')
xfm = wt_m(wave='db1')
ifm = iwt_m(wave='db1')

img_1 = plt.imread('0068_1_0.16.jpg')
# img_1 = pywt.data.camera()
# img_2 = pywt.data.ascent()
# img = np.stack([img_1, img_2], 0)
input = torch.tensor(img_1).float()
# input = input.reshape((1,400,400,3))
input = input.unsqueeze(0)
input = input.permute(0,3, 1, 2)
print(input.shape)
Y = xfm(input)
print(Y.shape)
print(10 * torch.log10(128**2 / F.mse_loss(Y[:,[1,2,3,5,6,7,9,10,11]], Y[:,[1,2,3,5,6,7,9,10,11]]*3)).item())
print(torch.min(Y[:,[0,4,8]]),torch.max(Y[:,[0,4,8]]))
print(torch.min(Y[:,[1,2,3,5,6,7,9,10,11]]),torch.max(Y[:,[1,2,3,5,6,7,9,10,11]]))
recon = ifm(Y)
print(10 * torch.log10(255**2 / F.mse_loss(input, recon)).item())
plt.subplot(2,3,1)
plt.imshow(input.squeeze(0).permute(1, 2, 0).int())
plt.subplot(2,3,2)
plt.imshow(Y[:,[0,4,8]].squeeze(0).permute(1, 2, 0).int())
plt.subplot(2,3,3)
plt.imshow(Y[:,[1,5,9]].squeeze(0).permute(1, 2, 0).int())
plt.subplot(2,3,4)
plt.imshow(Y[:,[2,6,10]].squeeze(0).permute(1, 2, 0).int())
plt.subplot(2,3,5)
plt.imshow(Y[:,[3,7,11]].squeeze(0).permute(1, 2, 0).int())
plt.subplot(2,3,6)
plt.imshow(recon.squeeze(0).permute(1, 2, 0).int())
plt.show()