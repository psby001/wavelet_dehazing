import torch.fft
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

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

img_1 = plt.imread('4.jpg')
input = torch.tensor(img_1).float()
# input = input[0].reshape(1,1,400,400)
input = input.permute(2, 0, 1)
input = input.unsqueeze(0)
print(input.shape)
fft_im,fft_amp, fft_pha = calculate_fft(input)
print(fft_amp.shape,fft_pha.shape)
recon = inverse_fft(fft_amp, fft_pha).real
# recon = torch.fft.ifft(fft_im)
plt.subplot(4,3,1)
plt.imshow(input.squeeze(0)[0].int(), cmap='gray')
plt.subplot(4,3,2)
plt.imshow(input.squeeze(0)[1].int(), cmap='gray')
plt.subplot(4,3,3)
plt.imshow(input.squeeze(0)[2].int(), cmap='gray')
plt.subplot(4,3,4)
plt.imshow(fft_amp.squeeze(0)[0].int(), cmap='gray')
plt.subplot(4,3,5)
plt.imshow(fft_amp.squeeze(0)[1].int(), cmap='gray')
plt.subplot(4,3,6)
plt.imshow(fft_amp.squeeze(0)[2].int(), cmap='gray')
plt.subplot(4,3,7)
plt.imshow(fft_pha.squeeze(0)[0].int(), cmap='gray')
plt.subplot(4,3,8)
plt.imshow(fft_pha.squeeze(0)[1].int(), cmap='gray')
plt.subplot(4,3,9)
plt.imshow(fft_pha.squeeze(0)[2].int(), cmap='gray')
plt.subplot(4,3,10)
plt.imshow(recon.squeeze(0)[0].int(), cmap='gray')
plt.subplot(4,3,11)
plt.imshow(recon.squeeze(0)[1].int(), cmap='gray')
plt.subplot(4,3,12)
plt.imshow(recon.squeeze(0)[2].int(), cmap='gray')
plt.show()