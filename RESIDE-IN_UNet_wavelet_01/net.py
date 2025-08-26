import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse


from SWT import SWTForward,SWTInverse
from wavelet import wt_m,iwt_m


sp_rate = 0.5
class resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.resconv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.act = nn.ReLU()
        self.act2 = nn.Sigmoid()
        # self.act = nn.Identity()
        # self.act2 = nn.Tanh()

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(out_channels, (out_channels // 2))
        self.fc2 = nn.Linear((out_channels // 2), out_channels)

    def forward(self, x):
        res = self.resconv(x)
        out1 = self.block1(x)

        out2 = self.block2(out1)
        out2_res = out2 + res
        out3 = self.block3(out2_res)

        out4 = self.block4(out3)
        # SE block part
        SEout = self.global_pool(out4)
        # permute to linear shape
        # (batch, channels, H, W) --> (batch, H, W, channels)
        SEout = SEout.permute(0, 2, 3, 1)
        SEout = self.fc1(SEout)
        SEout = self.act(SEout)
        SEout = self.fc2(SEout)
        SEout = self.act2(SEout)
        # recover to (batch, channels, H, W)
        SEout = SEout.permute(0, 3, 1, 2)

        out4 = out4 * SEout

        return out4 + out2_res


class down_sample_net(nn.Module):
    def __init__(self, in_channels=3,block =resblock):
        super(down_sample_net, self).__init__()
        # self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down1 = self.down_sample(3,64)
        self.down2 = self.down_sample(64,128)
        self.down3 = self.down_sample(128,256)
        self.down4 = self.down_sample(256,512)
        # # self.down = wt_m()
        # self.up = Interpolate(scale_factor=2, mode='bilinear')
        self.up_conv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = nn.ConvTranspose2d(128 ,64, kernel_size=2, stride=2)
        self.up_conv1 = nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2)
    def down_sample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        # x1 = self.down(x)
        # x2= self.down(x1)
        # x3 = self.down(x2)
        # x4 = self.down(x3)
        # x5 = self.up(x4)
        # x6 = self.up(x5)
        # x7 = self.up(x6)
        # x8 = self.up(x7)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.up_conv4(x4)
        x6 = self.up_conv3(x5)
        x7 = self.up_conv2(x6)
        x8 = self.up_conv1(x7)

        return x8

class decoder(nn.Module):
    def __init__(self, in_channels=[150,150,150,512],inner_channels = [32,64,128,256], out_channels=3,out_act = nn.ELU(),block = resblock):
        super(decoder, self).__init__()
        # self.xfm = wt_m(requires_grad=False)  # Accepts all wave types available to PyWavelets
        # self.ifm = iwt_m(requires_grad=False)

        self.decoder3 = block(in_channels[2] + inner_channels[3], inner_channels[3])
        self.decoder2 = block(in_channels[1] + inner_channels[2], inner_channels[2])
        # self.decoder1 = block(in_channels[0] + inner_channels[1], inner_channels[1])
        # self.decoder0 = block(3 + 32, 32)

        self.out = nn.Sequential(
                nn.Conv2d(in_channels[0] + inner_channels[1], out_channels, 1),
                nn.BatchNorm2d(out_channels),
                out_act
            )
        # self.out = nn.Sequential(
        #     nn.Conv2d(in_channels[0] + inner_channels[1], inner_channels[0], 1),
        #     nn.BatchNorm2d(inner_channels[0]),
        #     nn.ELU(),
        #     nn.Conv2d(inner_channels[0], out_channels, 1),
        #     # nn.ELU()
        #     out_act
        # )
        self.up_conv4 = nn.ConvTranspose2d(in_channels[3], inner_channels[3], kernel_size=2, stride=2)
        self.up_conv3 = nn.ConvTranspose2d(inner_channels[3], inner_channels[2], kernel_size=2, stride=2)
        self.up_conv2 = nn.ConvTranspose2d(inner_channels[2],  inner_channels[1], kernel_size=2, stride=2)
    def forward(self, enc1, enc2, enc3, enc4):
        dec4_up = self.up_conv4(enc4)
        # dec4_up = self.ifm(enc4)

        dec3_skip = enc3
        dec3_in = torch.cat((dec4_up, dec3_skip), dim=1)
        dec3_out = self.decoder3(dec3_in)
        dec3_up = self.up_conv3(dec3_out)
        # dec3_up = self.ifm(dec3_out)

        dec2_skip = enc2
        dec2_in = torch.cat((dec3_up, dec2_skip), dim=1)
        dec2_out = self.decoder2(dec2_in)
        dec2_up = self.up_conv2(dec2_out)
        # dec2_up = self.ifm(dec2_out)

        dec1_skip = enc1
        dec1_in = torch.cat((dec2_up, dec1_skip), dim=1)
        output = self.out(dec1_in)
        return output,dec2_out,dec3_out
    def reset_arg(self):
        self.decoder3.reset_arg()
        self.decoder2.reset_arg()

    def export_model(self):
        return decoder(model=self)

class decoder_lite1(nn.Module):
    def __init__(self, in_channels=[150,150,150,512],inner_channels = [32,64,128,256], out_channels=3,out_act = nn.ELU(),block = resblock,model = None):
        super(decoder_lite1, self).__init__()
        # self.xfm = wt_m(requires_grad=False)  # Accepts all wave types available to PyWavelets
        # self.ifm = iwt_m(requires_grad=False)
        # if model is None:

        # self.decoder3 = block(inner_channels[2], inner_channels[2])
        # self.decoder2 = block(in_channels[1] + inner_channels[2], inner_channels[2])
        # self.decoder1 = block(in_channels[0] + inner_channels[1], inner_channels[1])
        # self.decoder0 = block(3 + 32, 32)

        self.out = nn.Sequential(
                nn.Conv2d(inner_channels[0] + inner_channels[2], out_channels, 1),
                nn.BatchNorm2d(out_channels),
                out_act
            )

        self.fusion2_1 = nn.Sequential(
            nn.Conv2d(in_channels[1] + in_channels[0], inner_channels[0], 1),
            nn.BatchNorm2d(inner_channels[0]),
            nn.ReLU()
        )

        self.fusion3_4 = nn.Sequential(
            nn.Conv2d(in_channels[3]+in_channels[2], inner_channels[2], 1),
            nn.BatchNorm2d(inner_channels[2]),
            nn.ReLU()
        )
        self.up_conv2 = Interpolate(scale_factor=2.0, mode='bilinear')
        self.up_conv4 = Interpolate(scale_factor=4.0, mode='bilinear')
    def forward(self, enc1, enc2, enc3, enc4):
        dec4_up = self.up_conv2(enc4)

        out_3 = self.fusion3_4(torch.cat([enc3, dec4_up],dim=1))
        # dec4_up = self.ifm(enc4)

        dec2_up = self.up_conv2(enc2)
        out_1 = self.fusion2_1(torch.cat([enc1, dec2_up], dim=1))
        # dec3_in = torch.cat((dec4_up, dec3_skip), dim=1)
        # dec3_out = self.decoder3(dec3_in)
        # dec3_up = self.up_conv3(dec3_out)
        # dec3_up = self.ifm(dec3_out)
        output = self.out(torch.cat([out_1,self.up_conv4(out_3)],dim=1))

        return output,out_1,out_3
    def reset_arg(self):
        self.decoder3.reset_arg()
        self.decoder2.reset_arg()

    def export_model(self):
        return decoder(model=self)
class decoder_lite1_1(nn.Module):
    def __init__(self, in_channels=[150,150,150,512],inner_channels = [32,64,128,256], out_channels=3,out_act = nn.ELU(),block = resblock,model = None):
        super(decoder_lite1_1, self).__init__()
        # self.xfm = wt_m(requires_grad=False)  # Accepts all wave types available to PyWavelets
        # self.ifm = iwt_m(requires_grad=False)
        # if model is None:

        # self.decoder3 = block(inner_channels[2], inner_channels[2])
        # self.decoder2 = block(in_channels[1] + inner_channels[2], inner_channels[2])
        # self.decoder1 = block(in_channels[0] + inner_channels[1], inner_channels[1])
        # self.decoder0 = block(3 + 32, 32)

        self.out = nn.Sequential(
                nn.Conv2d(inner_channels[0] + inner_channels[2], out_channels, 1),
                nn.BatchNorm2d(out_channels),
                out_act
            )

        # self.fusion2_1 = nn.Sequential(
        #     nn.Conv2d(in_channels[1] + in_channels[0], inner_channels[0], 1),
        #     nn.BatchNorm2d(inner_channels[0]),
        #     nn.ReLU()
        # )
        self.fusion2_1 = block(in_channels[1] + in_channels[0], inner_channels[0])
        self.fusion3_4 = block(in_channels[3]+in_channels[2], inner_channels[2])


        self.up_conv2 = Interpolate(scale_factor=2.0, mode='bilinear')
        self.up_conv4 = Interpolate(scale_factor=4.0, mode='bilinear')
    def forward(self, enc1, enc2, enc3, enc4):
        dec4_up = self.up_conv2(enc4)

        out_3 = self.fusion3_4(torch.cat([enc3, dec4_up],dim=1))
        # dec4_up = self.ifm(enc4)

        dec2_up = self.up_conv2(enc2)
        out_1 = self.fusion2_1(torch.cat([enc1, dec2_up], dim=1))
        # dec3_in = torch.cat((dec4_up, dec3_skip), dim=1)
        # dec3_out = self.decoder3(dec3_in)
        # dec3_up = self.up_conv3(dec3_out)
        # dec3_up = self.ifm(dec3_out)
        output = self.out(torch.cat([out_1,self.up_conv4(out_3)],dim=1))

        return output,out_1,out_3

class decoder_lite1_2(nn.Module):
    def __init__(self, in_channels=[150,150,150,512],inner_channels = [32,64,128,256], out_channels=3,out_act = nn.ELU(),block = resblock,model = None):
        super(decoder_lite1_2, self).__init__()
        # self.xfm = wt_m(requires_grad=False)  # Accepts all wave types available to PyWavelets
        # self.ifm = iwt_m(requires_grad=False)
        # if model is None:

        # self.decoder3 = block(inner_channels[2], inner_channels[2])
        # self.decoder2 = block(in_channels[1] + inner_channels[2], inner_channels[2])
        # self.decoder1 = block(in_channels[0] + inner_channels[1], inner_channels[1])
        # self.decoder0 = block(3 + 32, 32)

        self.out = nn.Sequential(
                nn.Conv2d((inner_channels[0] + inner_channels[2]) // 2, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                out_act
            )

        self.fusion2_1 = nn.Sequential(
            nn.Conv2d(in_channels[1] + in_channels[0], inner_channels[0], 1),
            nn.BatchNorm2d(inner_channels[0]),
            nn.ReLU()
        )

        self.fusion3_4 = nn.Sequential(
            nn.Conv2d(in_channels[3]+in_channels[2], inner_channels[2], 1),
            nn.BatchNorm2d(inner_channels[2]),
            nn.ReLU()
        )

        self.fusion_all = block(inner_channels[0] + inner_channels[2], (inner_channels[0] + inner_channels[2]) // 2)
        self.up_conv2 = Interpolate(scale_factor=2., mode='bilinear')
        self.up_conv4 = Interpolate(scale_factor=4., mode='bilinear')
    def forward(self, enc1, enc2, enc3, enc4):
        dec4_up = self.up_conv2(enc4)

        out_3 = self.fusion3_4(torch.cat([enc3, dec4_up],dim=1))
        # dec4_up = self.ifm(enc4)

        dec2_up = self.up_conv2(enc2)
        out_1 = self.fusion2_1(torch.cat([enc1, dec2_up], dim=1))
        # dec3_in = torch.cat((dec4_up, dec3_skip), dim=1)
        # dec3_out = self.decoder3(dec3_in)
        # dec3_up = self.up_conv3(dec3_out)
        # dec3_up = self.ifm(dec3_out)
        output = self.out(self.fusion_all(torch.cat([out_1,self.up_conv4(out_3)],dim=1)))

        return output,out_1,out_3
    def reset_arg(self):
        self.decoder3.reset_arg()
        self.decoder2.reset_arg()

    def export_model(self):
        return decoder(model=self)

class decoder_lite2(nn.Module):
    def __init__(self, in_channels=[150,150,150,512],inner_channels = [32,64,128,256], out_channels=3,out_act = nn.ELU(),block = resblock,model = None):
        super(decoder_lite2, self).__init__()
        # self.xfm = wt_m(requires_grad=False)  # Accepts all wave types available to PyWavelets
        # self.ifm = iwt_m(requires_grad=False)
        # if model is None:

        # self.decoder3 = block(inner_channels[2], inner_channels[2])
        # self.decoder2 = block(in_channels[1] + inner_channels[2], inner_channels[2])
        # self.decoder1 = block(in_channels[0] + inner_channels[1], inner_channels[1])
        # self.decoder0 = block(3 + 32, 32)

        self.out = nn.Sequential(
                nn.Conv2d(inner_channels[0] + inner_channels[1], out_channels, 1),
                nn.BatchNorm2d(out_channels),
                out_act
            )

        self.fusion4_1 = nn.Sequential(
            nn.Conv2d(in_channels[3] + in_channels[0], inner_channels[0], 1),
            nn.BatchNorm2d(inner_channels[0]),
            nn.ReLU()
        )

        self.fusion3_2 = nn.Sequential(
            nn.Conv2d(in_channels[2]+in_channels[1], inner_channels[1], 1),
            nn.BatchNorm2d(inner_channels[1]),
            nn.ReLU()
        )
        self.up_conv2 = Interpolate(scale_factor=2., mode='bilinear')
        self.up_conv8 = Interpolate(scale_factor=8., mode='bilinear')
    def forward(self, enc1, enc2, enc3, enc4):
        dec4_up = self.up_conv8(enc4)

        out_1 = self.fusion4_1(torch.cat([enc1, dec4_up],dim=1))
        # dec4_up = self.ifm(enc4)

        dec3_up = self.up_conv2(enc3)
        out_2 = self.fusion3_2(torch.cat([enc2, dec3_up], dim=1))
        # dec3_in = torch.cat((dec4_up, dec3_skip), dim=1)
        # dec3_out = self.decoder3(dec3_in)
        # dec3_up = self.up_conv3(dec3_out)
        # dec3_up = self.ifm(dec3_out)
        output = self.out(torch.cat([out_1,self.up_conv2(out_2)],dim=1))

        return output,out_1,out_2
    def reset_arg(self):
        self.decoder3.reset_arg()
        self.decoder2.reset_arg()

    def export_model(self):
        return decoder(model=self)

class decoder_lite2_1(nn.Module):
    def __init__(self, in_channels=[150,150,150,512],inner_channels = [32,64,128,256], out_channels=3,out_act = nn.ELU(),block = resblock,model = None):
        super(decoder_lite2_1, self).__init__()
        # self.xfm = wt_m(requires_grad=False)  # Accepts all wave types available to PyWavelets
        # self.ifm = iwt_m(requires_grad=False)
        # if model is None:

        # self.decoder3 = block(inner_channels[2], inner_channels[2])
        # self.decoder2 = block(in_channels[1] + inner_channels[2], inner_channels[2])
        # self.decoder1 = block(in_channels[0] + inner_channels[1], inner_channels[1])
        # self.decoder0 = block(3 + 32, 32)

        self.out = nn.Sequential(
                nn.Conv2d(inner_channels[0] + inner_channels[1], out_channels, 1),
                nn.BatchNorm2d(out_channels),
                out_act
            )

        self.fusion4_1 = block(in_channels[3] + in_channels[0], inner_channels[0])

        self.fusion3_2 = block(in_channels[2]+in_channels[1], inner_channels[1])

        self.up_conv2 = Interpolate(scale_factor=2., mode='bilinear')
        self.up_conv8 = Interpolate(scale_factor=8., mode='bilinear')
    def forward(self, enc1, enc2, enc3, enc4):
        dec4_up = self.up_conv8(enc4)

        out_1 = self.fusion4_1(torch.cat([enc1, dec4_up],dim=1))
        # dec4_up = self.ifm(enc4)

        dec3_up = self.up_conv2(enc3)
        out_2 = self.fusion3_2(torch.cat([enc2, dec3_up], dim=1))
        # dec3_in = torch.cat((dec4_up, dec3_skip), dim=1)
        # dec3_out = self.decoder3(dec3_in)
        # dec3_up = self.up_conv3(dec3_out)
        # dec3_up = self.ifm(dec3_out)
        output = self.out(torch.cat([out_1,self.up_conv2(out_2)],dim=1))

        return output,out_1,out_2
    def reset_arg(self):
        self.decoder3.reset_arg()
        self.decoder2.reset_arg()

    def export_model(self):
        return decoder(model=self)

class decoder_lite2_2(nn.Module):
    def __init__(self, in_channels=[150,150,150,512],inner_channels = [32,64,128,256], out_channels=3,out_act = nn.ELU(),block = resblock,model = None):
        super(decoder_lite2_2, self).__init__()
        # self.xfm = wt_m(requires_grad=False)  # Accepts all wave types available to PyWavelets
        # self.ifm = iwt_m(requires_grad=False)
        # if model is None:

        # self.decoder3 = block(inner_channels[2], inner_channels[2])
        # self.decoder2 = block(in_channels[1] + inner_channels[2], inner_channels[2])
        # self.decoder1 = block(in_channels[0] + inner_channels[1], inner_channels[1])
        # self.decoder0 = block(3 + 32, 32)

        self.out = nn.Sequential(
                nn.Conv2d((inner_channels[0] + inner_channels[1])//2, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                out_act
            )

        self.fusion4_1 = nn.Sequential(
            nn.Conv2d(in_channels[3] + in_channels[0], inner_channels[0], 1),
            nn.BatchNorm2d(inner_channels[0]),
            nn.ReLU()
        )

        self.fusion3_2 = nn.Sequential(
            nn.Conv2d(in_channels[2]+in_channels[1], inner_channels[1], 1),
            nn.BatchNorm2d(inner_channels[1]),
            nn.ReLU()
        )

        self.fusion_all = block(inner_channels[0] + inner_channels[1], (inner_channels[0] + inner_channels[1])//2)
        self.up_conv2 = Interpolate(scale_factor=2., mode='bilinear')
        self.up_conv8 = Interpolate(scale_factor=8., mode='bilinear')
    def forward(self, enc1, enc2, enc3, enc4):
        dec4_up = self.up_conv8(enc4)

        out_1 = self.fusion4_1(torch.cat([enc1, dec4_up],dim=1))
        # dec4_up = self.ifm(enc4)

        dec3_up = self.up_conv2(enc3)
        out_2 = self.fusion3_2(torch.cat([enc2, dec3_up], dim=1))
        # dec3_in = torch.cat((dec4_up, dec3_skip), dim=1)
        # dec3_out = self.decoder3(dec3_in)
        # dec3_up = self.up_conv3(dec3_out)
        # dec3_up = self.ifm(dec3_out)
        output = self.out(self.fusion_all(torch.cat([out_1,self.up_conv2(out_2)],dim=1)))

        return output,out_1,out_2
    def reset_arg(self):
        self.decoder3.reset_arg()
        self.decoder2.reset_arg()

    def export_model(self):
        return decoder(model=self)

class decoder_lite3(nn.Module):
    def __init__(self, in_channels=[150,150,150,512],inner_channels = [32,64,128,256], out_channels=3,out_act = nn.ELU(),block = resblock,model = None):
        super(decoder_lite3, self).__init__()
        # self.xfm = wt_m(requires_grad=False)  # Accepts all wave types available to PyWavelets
        # self.ifm = iwt_m(requires_grad=False)
        # if model is None:

        # self.decoder3 = block(inner_channels[2], inner_channels[2])
        # self.decoder2 = block(in_channels[1] + inner_channels[2], inner_channels[2])
        # self.decoder1 = block(in_channels[0] + inner_channels[1], inner_channels[1])
        # self.decoder0 = block(3 + 32, 32)

        self.out = nn.Sequential(
                nn.Conv2d(inner_channels[0] + inner_channels[1], out_channels, 1),
                nn.BatchNorm2d(out_channels),
                out_act
            )

        self.fusion3_1 = nn.Sequential(
            nn.Conv2d(in_channels[2] + in_channels[0], inner_channels[0], 1),
            nn.BatchNorm2d(inner_channels[0]),
            nn.ReLU()
        )

        self.fusion4_2 = nn.Sequential(
            nn.Conv2d(in_channels[3]+in_channels[1], inner_channels[1], 1),
            nn.BatchNorm2d(inner_channels[1]),
            nn.ReLU()
        )
        self.up_conv2 = Interpolate(scale_factor=2., mode='bilinear')
        self.up_conv4 = Interpolate(scale_factor=4., mode='bilinear')
    def forward(self, enc1, enc2, enc3, enc4):
        dec3_up = self.up_conv4(enc3)

        out_1 = self.fusion3_1(torch.cat([enc1, dec3_up],dim=1))
        # dec4_up = self.ifm(enc4)

        dec4_up = self.up_conv4(enc4)
        out_2 = self.fusion4_2(torch.cat([enc2, dec4_up], dim=1))
        # dec3_in = torch.cat((dec4_up, dec3_skip), dim=1)
        # dec3_out = self.decoder3(dec3_in)
        # dec3_up = self.up_conv3(dec3_out)
        # dec3_up = self.ifm(dec3_out)
        output = self.out(torch.cat([out_1,self.up_conv2(out_2)],dim=1))

        return output,out_1,out_2
    def reset_arg(self):
        self.decoder3.reset_arg()
        self.decoder2.reset_arg()

    def export_model(self):
        return decoder(model=self)

class decoder_lite3_1(nn.Module):
    def __init__(self, in_channels=[150,150,150,512],inner_channels = [32,64,128,256], out_channels=3,out_act = nn.ELU(),block = resblock,model = None):
        super(decoder_lite3_1, self).__init__()
        # self.xfm = wt_m(requires_grad=False)  # Accepts all wave types available to PyWavelets
        # self.ifm = iwt_m(requires_grad=False)
        # if model is None:

        # self.decoder3 = block(inner_channels[2], inner_channels[2])
        # self.decoder2 = block(in_channels[1] + inner_channels[2], inner_channels[2])
        # self.decoder1 = block(in_channels[0] + inner_channels[1], inner_channels[1])
        # self.decoder0 = block(3 + 32, 32)

        self.out = nn.Sequential(
                nn.Conv2d(inner_channels[0] + inner_channels[1], out_channels, 1),
                nn.BatchNorm2d(out_channels),
                out_act
            )

        self.fusion3_1 = block(in_channels[2] + in_channels[0], inner_channels[0])

        self.fusion4_2 = block(in_channels[3]+in_channels[1], inner_channels[1])
        self.up_conv2 = Interpolate(scale_factor=2., mode='bilinear')
        self.up_conv4 = Interpolate(scale_factor=4., mode='bilinear')
    def forward(self, enc1, enc2, enc3, enc4):
        dec3_up = self.up_conv4(enc3)

        out_1 = self.fusion3_1(torch.cat([enc1, dec3_up],dim=1))
        # dec4_up = self.ifm(enc4)

        dec4_up = self.up_conv4(enc4)
        out_2 = self.fusion4_2(torch.cat([enc2, dec4_up], dim=1))
        # dec3_in = torch.cat((dec4_up, dec3_skip), dim=1)
        # dec3_out = self.decoder3(dec3_in)
        # dec3_up = self.up_conv3(dec3_out)
        # dec3_up = self.ifm(dec3_out)
        output = self.out(torch.cat([out_1,self.up_conv2(out_2)],dim=1))

        return output,out_1,out_2
    def reset_arg(self):
        self.decoder3.reset_arg()
        self.decoder2.reset_arg()

    def export_model(self):
        return decoder(model=self)

class decoder_lite3_2(nn.Module):
    def __init__(self, in_channels=[150,150,150,512],inner_channels = [32,64,128,256], out_channels=3,out_act = nn.ELU(),block = resblock,model = None):
        super(decoder_lite3_2, self).__init__()
        # self.xfm = wt_m(requires_grad=False)  # Accepts all wave types available to PyWavelets
        # self.ifm = iwt_m(requires_grad=False)
        # if model is None:

        # self.decoder3 = block(inner_channels[2], inner_channels[2])
        # self.decoder2 = block(in_channels[1] + inner_channels[2], inner_channels[2])
        # self.decoder1 = block(in_channels[0] + inner_channels[1], inner_channels[1])
        # self.decoder0 = block(3 + 32, 32)

        self.out = nn.Sequential(
                nn.Conv2d((inner_channels[0] + inner_channels[1])//2, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                out_act
            )

        self.fusion3_1 = nn.Sequential(
            nn.Conv2d(in_channels[2] + in_channels[0], inner_channels[0], 1),
            nn.BatchNorm2d(inner_channels[0]),
            nn.ReLU()
        )

        self.fusion4_2 = nn.Sequential(
            nn.Conv2d(in_channels[3]+in_channels[1], inner_channels[1], 1),
            nn.BatchNorm2d(inner_channels[1]),
            nn.ReLU()
        )

        self.fusion_all = block(inner_channels[0] + inner_channels[1], (inner_channels[0] + inner_channels[1])//2)
        self.up_conv2 = Interpolate(scale_factor=2., mode='bilinear')
        self.up_conv4 = Interpolate(scale_factor=4., mode='bilinear')
    def forward(self, enc1, enc2, enc3, enc4):
        dec3_up = self.up_conv4(enc3)

        out_1 = self.fusion3_1(torch.cat([enc1, dec3_up],dim=1))
        # dec4_up = self.ifm(enc4)

        dec4_up = self.up_conv4(enc4)
        out_2 = self.fusion4_2(torch.cat([enc2, dec4_up], dim=1))
        # dec3_in = torch.cat((dec4_up, dec3_skip), dim=1)
        # dec3_out = self.decoder3(dec3_in)
        # dec3_up = self.up_conv3(dec3_out)
        # dec3_up = self.ifm(dec3_out)
        output = self.out(self.fusion_all(torch.cat([out_1,self.up_conv2(out_2)],dim=1)))

        return output,out_1,out_2
    def reset_arg(self):
        self.decoder3.reset_arg()
        self.decoder2.reset_arg()

    def export_model(self):
        return decoder(model=self)

class decoder_lite4(nn.Module):
    def __init__(self, in_channels=[150,150,150,512],inner_channels = [32,64,128,256], out_channels=3,out_act = nn.ELU(),block = resblock,model = None):
        super(decoder_lite4, self).__init__()
        # self.xfm = wt_m(requires_grad=False)  # Accepts all wave types available to PyWavelets
        # self.ifm = iwt_m(requires_grad=False)
        # if model is None:

        # self.decoder = block(inner_channels[2], inner_channels[2])
        # self.decoder2 = block(in_channels[1] + inner_channels[2], inner_channels[2])
        # self.decoder1 = block(in_channels[0] + inner_channels[1], inner_channels[1])
        # self.decoder0 = block(3 + 32, 32)

        self.out = nn.Sequential(
                nn.Conv2d(inner_channels[0] + inner_channels[1]+ in_channels[3], out_channels, 1),
                nn.BatchNorm2d(out_channels),
                out_act
            )

        # self.fusion2_1 = nn.Sequential(
        #     nn.Conv2d(in_channels[1] + in_channels[0], inner_channels[0], 1),
        #     nn.BatchNorm2d(inner_channels[0]),
        #     nn.ReLU()
        # )
        self.fusion2_1 = block(in_channels[1] + in_channels[0], inner_channels[0])
        self.fusion2_3 = block(in_channels[1]+in_channels[2], inner_channels[1])


        self.up_conv2 = Interpolate(scale_factor=2., mode='bilinear')
        self.up_conv8 = Interpolate(scale_factor=8., mode='bilinear')
    def forward(self, enc1, enc2, enc3, enc4):
        dec3_up = self.up_conv2(enc3)

        out_2_3 = self.fusion2_3(torch.cat([enc2, dec3_up],dim=1))
        # dec4_up = self.ifm(enc4)

        dec2_up = self.up_conv2(enc2)
        out_2_1 = self.fusion2_1(torch.cat([enc1, dec2_up], dim=1))
        # dec3_in = torch.cat((dec4_up, dec3_skip), dim=1)
        # dec3_out = self.decoder3(dec3_in)
        # dec3_up = self.up_conv3(dec3_out)
        # dec3_up = self.ifm(dec3_out)
        output = self.out(torch.cat([out_2_1,self.up_conv2(out_2_3),self.up_conv8(enc4)],dim=1))

        return output,out_2_1,out_2_3

class decoder_lite5(nn.Module):
    def __init__(self, in_channels=[150,150,150,512],inner_channels = [32,64,128,256], out_channels=3,out_act = nn.ELU(),block = resblock,model = None):
        super(decoder_lite5, self).__init__()
        # self.xfm = wt_m(requires_grad=False)  # Accepts all wave types available to PyWavelets
        # self.ifm = iwt_m(requires_grad=False)
        # if model is None:

        self.decoder = block(in_channels[0] + inner_channels[1] + in_channels[3], inner_channels[0])
        # self.decoder2 = block(in_channels[1] + inner_channels[2], inner_channels[2])
        # self.decoder1 = block(in_channels[0] + inner_channels[1], inner_channels[1])
        # self.decoder0 = block(3 + 32, 32)

        self.out = nn.Sequential(
                nn.Conv2d(inner_channels[0], out_channels, 1),
                nn.BatchNorm2d(out_channels),
                out_act
            )

        # self.fusion2_1 = nn.Sequential(
        #     nn.Conv2d(in_channels[1] + in_channels[0], inner_channels[0], 1),
        #     nn.BatchNorm2d(inner_channels[0]),
        #     nn.ReLU()
        # )
        self.fusion2_3 = block(in_channels[1]+in_channels[2], inner_channels[1])


        self.up_conv2 = Interpolate(scale_factor=2., mode='bilinear')
        self.up_conv8 = Interpolate(scale_factor=8., mode='bilinear')
    def forward(self, enc1, enc2, enc3, enc4):
        dec3_up = self.up_conv2(enc3)

        out_2_3 = self.fusion2_3(torch.cat([enc2, dec3_up],dim=1))
        # dec4_up = self.ifm(enc4)
        # dec3_in = torch.cat((dec4_up, dec3_skip), dim=1)
        # dec3_out = self.decoder3(dec3_in)
        # dec3_up = self.up_conv3(dec3_out)
        # dec3_up = self.ifm(dec3_out)
        out_1 = self.decoder(torch.cat([enc1,self.up_conv2(out_2_3),self.up_conv8(enc4)], dim=1))


        output = self.out(out_1)

        return output,out_1,out_2_3

class UNet(nn.Module):
    def __init__(self,in_channels=3,out_channels=3,block = resblock):
        super(UNet, self).__init__()

        # 下採樣層
        self.encoder = FSnet(in_channels,block =block)

        # 中間層

        # 上採樣層
        self.decoder3 = block(406, 256)
        self.decoder2 = block(278, 128)
        self.decoder1 = block(214, 64)
        self.out = nn.Sequential(
            nn.Conv2d(64, out_channels, 1),
            # nn.Sigmoid()
        )

        self.up_conv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)



    def forward(self, x,decoder = True):
        enc1, enc2, enc3, enc4 = self.encoder(x)
        if decoder:
            dec4_up = self.up_conv4(enc4)

            dec3_skip = enc3
            dec3_in = torch.cat((dec4_up, dec3_skip), dim=1)
            dec3_out = self.decoder3(dec3_in)
            dec3_up = self.up_conv3(dec3_out)

            dec2_skip = enc2
            dec2_in = torch.cat((dec3_up, dec2_skip), dim=1)
            dec2_out = self.decoder2(dec2_in)
            dec2_up = self.up_conv2(dec2_out)

            dec1_skip = enc1
            dec1_in = torch.cat((dec2_up, dec1_skip), dim=1)
            output1 = self.decoder1(dec1_in)
            output1 = self.out(output1)
            output1 = torch.clamp(output1, 0, 1.0)

            # return output1
            return (output1,enc1, enc2, enc3, enc4)
        else:
            return (enc1, enc2, enc3, enc4)

    def forward_stage_1(self, x):
        enc1, enc2, enc3, enc4 = self.encoder(x)
        return (enc1, enc2, enc3, enc4)
    def forward_stage_2(self,enc1, enc2, enc3, enc4):
        dec4_up = self.up_conv4(enc4)

        dec3_skip = enc3
        dec3_in = torch.cat((dec4_up, dec3_skip), dim=1)
        dec3_out = self.decoder3(dec3_in)
        dec3_up = self.up_conv3(dec3_out)

        dec2_skip = enc2
        dec2_in = torch.cat((dec3_up, dec2_skip), dim=1)
        dec2_out = self.decoder2(dec2_in)
        dec2_up = self.up_conv2(dec2_out)

        dec1_skip = enc1
        dec1_in = torch.cat((dec2_up, dec1_skip), dim=1)
        output1 = self.decoder1(dec1_in)
        output1 = self.out(output1)
        output1 = torch.clamp(output1, 0, 1.0)

        # return output1
        return output1

class mspblock(nn.Module):
    def __init__(self, in_channels, out_channels,a=sp_rate):
        super().__init__()
        self.resconv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.channel1 = round(out_channels*a)
        self.channel2 = round(out_channels*(a**2))

        self.block1 = nn.Sequential(
            nn.Conv2d(self.channel1, self.channel1, 3, padding=1,groups=self.channel1),
            # nn.Conv2d(self.channel1, self.channel1, 3, padding=1),
            nn.BatchNorm2d(self.channel1),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(self.channel2, self.channel2, 3, padding=1),
            # nn.Conv2d(self.channel2, self.channel2, 3, padding=1),
            nn.BatchNorm2d(self.channel2),
            nn.ReLU()
        )

        # self.block2_fu = nn.Sequential(
        #     nn.Conv2d(self.channel1, self.channel1, 1),
        #     nn.BatchNorm2d(self.channel1),
        #     nn.ReLU()
        # )

        self.block3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        res = self.resconv(x)
        B,C,H,W = res.shape
        part1,part2 = res[:,:self.channel1],res[:,self.channel1:]
        out1 = self.block1(part1)
        out1_1,out1_2 = out1[:,:self.channel2],out1[:,self.channel2:]

        out1_1 = self.block2(out1_1)
        # out2 = self.block2_fu(torch.concat((out1_1,out1_2),dim=1))
        out2 = torch.concat((out1_1,out1_2),dim=1)

        out2_res = torch.concat([part2,out2],dim=1)
        out3 = self.block3(out2_res)

        return out3

class rescspblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.resconv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        self.block1 = nn.Sequential(
            nn.Conv2d(out_channels//2, out_channels//2, 3, padding=1),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels//2, out_channels//2, 3, padding=1),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.act = nn.ReLU()
        # self.act = nn.Identity()
        self.act2 = nn.Sigmoid()
        # self.act2 = nn.Tanh()

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(out_channels, (out_channels // 2))
        self.fc2 = nn.Linear((out_channels // 2), out_channels)

    def forward(self, x):
        res = self.resconv(x)
        B,C,H,W = res.shape
        part1,part2 = res[:,:C//2],res[:,C//2:]
        out1 = self.block1(part1)

        out2 = self.block2(out1)
        out2_res = torch.concat([part2,out2],dim=1)
        out3 = self.block3(out2_res)

        out4 = self.block4(out3)
        # SE block part
        SEout = self.global_pool(out4)
        # permute to linear shape
        # (batch, channels, H, W) --> (batch, H, W, channels)
        SEout = SEout.permute(0, 2, 3, 1)
        SEout = self.fc1(SEout)
        SEout = self.act(SEout)
        SEout = self.fc2(SEout)
        SEout = self.act2(SEout)
        # recover to (batch, channels, H, W)
        SEout = SEout.permute(0, 3, 1, 2)

        out4 = out4 * SEout

        return out4

class resmspblock(nn.Module):
    def __init__(self, in_channels, out_channels,a=sp_rate):
        super().__init__()
        self.resconv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.channel1 = round(out_channels*a)
        self.channel2 = round(out_channels*(a**2))

        self.block1 = nn.Sequential(
            nn.Conv2d(self.channel1, self.channel1, 3, padding=1,groups=self.channel1),
            # nn.Conv2d(self.channel1, self.channel1, 3, padding=1),
            nn.BatchNorm2d(self.channel1),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(self.channel2, self.channel2, 3, padding=1),
            # nn.Conv2d(self.channel2, self.channel2, 3, padding=1),
            nn.BatchNorm2d(self.channel2),
            nn.ReLU()
        )

        # self.block2_fu = nn.Sequential(
        #     nn.Conv2d(self.channel1, self.channel1, 1),
        #     nn.BatchNorm2d(self.channel1),
        #     nn.ReLU()
        # )

        self.block3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.act = nn.ReLU()
        # self.act = nn.Identity()
        self.act2 = nn.Sigmoid()
        # self.act2 = nn.Tanh()

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(out_channels, (out_channels // 2))
        self.fc2 = nn.Linear((out_channels // 2), out_channels)

    def forward(self, x):
        res = self.resconv(x)
        B,C,H,W = res.shape
        part1,part2 = res[:,:self.channel1],res[:,self.channel1:]
        out1 = self.block1(part1)
        out1_1,out1_2 = out1[:,:self.channel2],out1[:,self.channel2:]

        out1_1 = self.block2(out1_1)
        # out2 = self.block2_fu(torch.concat((out1_1,out1_2),dim=1))
        out2 = torch.concat((out1_1,out1_2),dim=1)

        out2_res = torch.concat([part2,out2],dim=1)
        out3 = self.block3(out2_res)

        out4 = self.block4(out3)
        # SE block part
        SEout = self.global_pool(out4)
        # permute to linear shape
        # (batch, channels, H, W) --> (batch, H, W, channels)
        SEout = SEout.permute(0, 2, 3, 1)
        SEout = self.fc1(SEout)
        SEout = self.act(SEout)
        SEout = self.fc2(SEout)
        SEout = self.act2(SEout)
        # recover to (batch, channels, H, W)
        SEout = SEout.permute(0, 3, 1, 2)

        out4 = out4 * SEout

        return out4

class shuffer_v2(nn.Module):
    def __init__(self, level):
        super().__init__()
        self.level = level
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ags = []
        for i in range(level):
            f= i
            b = level - i - 1
            ags = [0]
            n = 1
            while n < level:
                if f > 0:
                    ags = [n] + ags
                    n +=1
                    f -= 1
                if b > 0:
                    ags = ags + [n]
                    n += 1
                    b -= 1
            self.ags.append(ags)
        self.ags = np.array(self.ags).T.tolist()
    def forward(self, levels,mode = 'bilinear'):
        out_levels =[]
        sort_ags = []
        for L in levels:
            B,C,H,W = L.shape
            sort_ags.append(torch.argsort(self.gap(L).reshape((B,C)), dim=1))
        for i in range(self.level):
            channels = levels[i].shape[1] // self.level
            out_level = []
            for j in range(self.level):
                # l = torch.cat([levels[i][n, sort_ags[i][[n], channels * j:channels * (j + 1)]] for n in range(levels[i].shape[0])], dim=0)
                l = torch.cat([levels[i][n, sort_ags[i][[n], channels * self.ags[i][j]:channels * (self.ags[i][j] + 1)]] for n in range(levels[i].shape[0])], dim=0)
                out_level.append(F.interpolate(l, scale_factor=float(1/(2**(j-i))), mode=mode))
            out_levels.append(out_level)
        output = []
        for i in range(self.level):
            output.append(torch.cat([out_levels[j][i] for j in range(self.level)], dim=1))
        return output

class cubic_attention(nn.Module):
    def __init__(self, dim, group, dilation, kernel) -> None:
        super().__init__()

        self.H_spatial_att = spatial_strip_att(dim, dilation=dilation, group=group, kernel=kernel)
        self.W_spatial_att = spatial_strip_att(dim, dilation=dilation, group=group, kernel=kernel, H=False)
        self.fushion = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1),
            nn.BatchNorm2d(dim),
        )
        self.gamma = nn.Parameter(torch.zeros(dim,1,1))
        self.beta = nn.Parameter(torch.ones(dim,1,1))

    def forward(self, x):
        out1 = self.H_spatial_att(x)
        out2 = self.W_spatial_att(out1)
        # out = torch.concat((out1,out2),dim=1)
        # out = self.fushion(out)


        # return self.gamma * x + out2 * self.beta
        return out2

class cubic_attention_2(nn.Module):
    def __init__(self, dim, kernel) -> None:
        super().__init__()

        self.H_spatial_att = spatial_strip_att_2(dim, kernel=kernel)
        self.W_spatial_att = spatial_strip_att_2(dim, kernel=kernel, H=False)
        # self.fushion = nn.Sequential(
        #     nn.Conv2d(dim*2, dim, 1),
        #     nn.BatchNorm2d(dim),
        # )
        self.gamma = nn.Parameter(torch.zeros(dim,1,1))
        self.beta = nn.Parameter(torch.ones(dim,1,1))

    def forward(self, x):
        out1 = self.H_spatial_att(x)
        out2 = self.W_spatial_att(x)
        out = torch.concat((out1,out2),dim=1)
        # out = self.fushion(out)


        # return self.gamma * out1 + out2 * self.beta
        return out

class spatial_strip_att_2(nn.Module):
    def __init__(self, dim, kernel=3, H=True) -> None:
        super().__init__()

        self.k = kernel
        self.dim = dim
        self.kernel = (1, kernel) if H else (kernel, 1)
        self.padding = (kernel//2, 1) if H else (1, kernel//2)
        pad = (kernel - 1) // 2
        self.pad = nn.ReflectionPad2d((pad, pad, 0, 0)) if H else nn.ReflectionPad2d((0, 0, pad, pad))
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, self.kernel,groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )

        self.act = nn.ReLU()
        # self.act = nn.Identity()
        self.act2 = nn.Sigmoid()
        # self.act2 = nn.Tanh()

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(dim, (dim // 2))
        self.fc2 = nn.Linear((dim // 2), dim)

    def forward(self, x):
        out = self.block(self.pad(x))
        # SE block part
        w = self.global_pool(out)
        # permute to linear shape
        # (batch, channels, H, W) --> (batch, H, W, channels)
        w = w.permute(0, 2, 3, 1)
        w = self.fc1(w)
        w = self.act(w)
        w = self.fc2(w)
        w = self.act2(w)
        # recover to (batch, channels, H, W)
        w = w.permute(0, 3, 1, 2)

        out = out * w

        return out
class spatial_strip_att(nn.Module):
    def __init__(self, dim, kernel=3, dilation=1, group=2, H=True) -> None:
        super().__init__()

        self.k = kernel
        self.dim = dim
        pad = dilation*(kernel-1) // 2
        self.kernel = (1, kernel) if H else (kernel, 1)
        self.padding = (kernel//2, 1) if H else (1, kernel//2)
        self.dilation = dilation
        self.group = group
        self.pad = nn.ReflectionPad2d((pad, pad, 0, 0)) if H else nn.ReflectionPad2d((0, 0, pad, pad))
        self.conv = nn.Conv2d(dim, dim*kernel*self.group, kernel_size=1, stride=1, bias=False)
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.filter_act = nn.Tanh()
        self.inside_all = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)
        self.lamb_l = nn.Parameter(torch.zeros(dim), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(dim), requires_grad=True)
        gap_kernel = (None,1) if H else (1, None)
        self.gap = nn.AdaptiveAvgPool2d(gap_kernel)

    def forward(self, x):
        identity_input = x.clone()
        filter = self.ap(x)
        filter = self.conv(filter)
        n, c, h, w = x.shape
        # x = F.unfold(self.pad(x), kernel_size=self.kernel, dilation=self.dilation).reshape(n, self.group, c//self.group, self.k, h*w)
        x = x.reshape(1,n*c, h, w)
        n, c1, p, q = filter.shape
        filter = filter.reshape(n*self.dim,1, self.kernel[0],self.kernel[1])
        out = F.conv2d(self.pad(x),filter,groups=self.dim*n)
        out = out.reshape((n, c, h, w))
        # filter = self.filter_act(filter)
        # out = torch.sum(x * filter, dim=3).reshape(n, c, h, w)

        # out_low = out * (self.inside_all + 1.) - self.inside_all * self.gap(identity_input)
        # out_low = out_low * self.lamb_l[None,:,None,None]
        # out_high = identity_input * (self.lamb_h[None,:,None,None]+1.)
        return out

        # return out_low + out_high

class resmspblock_sp(nn.Module):
    def __init__(self, in_channels, out_channels,a=sp_rate):
        super().__init__()
        self.resconv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.channel1 = round(out_channels*a)
        self.channel2 = round(out_channels*(a**2))
        # self.sort_arg = nn.Parameter(torch.tensor(range(out_channels)),requires_grad=False)
        # self.sort_arg2 = nn.Parameter(torch.tensor(range(self.channel1)),requires_grad=False)
        # self.gap = nn.AdaptiveAvgPool2d(1)
        # self.sort_tensor = torch.zeros((out_channels))
        # self.sort_tensor2 = torch.zeros((self.channel1))

        self.block1 = nn.Sequential(
            nn.Conv2d(self.channel1, self.channel1, 3, padding=1,groups=self.channel1),
            # nn.Conv2d(self.channel1, self.channel1, 3, padding=1),
            nn.BatchNorm2d(self.channel1),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(self.channel2, self.channel2, 3, padding=1),
            # nn.Conv2d(self.channel2, self.channel2, 3, padding=1),
            nn.BatchNorm2d(self.channel2),
            nn.ReLU()
        )

        # self.block2_fu = nn.Sequential(
        #     nn.Conv2d(self.channel1, self.channel1, 1),
        #     nn.BatchNorm2d(self.channel1),
        #     nn.ReLU()
        # )

        self.block3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        # self.block4 = nn.Sequential(
        #     nn.Conv2d(out_channels, out_channels, 3, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU()
        # )
        # self.strip_att = cubic_attention(out_channels, group=1, dilation=1, kernel=3)
        self.strip_att = cubic_attention_2(self.channel1, kernel=3)

        self.fushion = nn.Sequential(
            nn.Conv2d((out_channels - self.channel1) + self.channel1*2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        res = self.resconv(x)
        B,C,H,W = res.shape
        # device = res.device
        # self.sort_tensor = self.sort_tensor.to(device)
        # self.sort_tensor = self.sort_tensor + torch.mean(self.gap(res).reshape((B, C)),dim=0)
        # res = res[:,self.sort_arg]
        part1,part2 = res[:,:self.channel1],res[:,self.channel1:]
        out1 = self.block1(part1)
        # self.sort_tensor2 = self.sort_tensor2.to(device)
        # self.sort_tensor2 = self.sort_tensor2 + torch.mean(self.gap(out1).reshape(out1.shape[:2]), dim=0)
        # out1 = out1[:, self.sort_arg2]
        out1_1,out1_2 = out1[:,:self.channel2],out1[:,self.channel2:]

        out1_1 = self.block2(out1_1)
        # out2 = self.block2_fu(torch.concat((out1_1,out1_2),dim=1))
        out2 = torch.concat((out1_1,out1_2),dim=1)

        out2_res = torch.concat([part2,out2],dim=1)
        out3 = self.block3(out2_res)

        out3_part1,out3_part2 = out3[:,:self.channel1],out3[:,self.channel1:]
        out4 = self.fushion(torch.cat((self.strip_att(out3_part1),out3_part2),dim=1))
        # out4 = self.fushion(torch.cat((self.strip_att(out3_part1),self.block4(out3_part2)),dim=1))

        #
        # return out4 + out3
        return out4
        # return out3

    # def reset_arg(self):
    #     self.sort_arg = nn.Parameter(torch.argsort(self.sort_tensor),requires_grad=False)
    #     self.sort_arg2 = nn.Parameter(torch.argsort(self.sort_tensor2),requires_grad=False)
    #     self.sort_tensor = self.sort_tensor * 0
    #     self.sort_tensor2 = self.sort_tensor2 * 0

class resmspblock_sp_v1(nn.Module):
    def __init__(self, in_channels, out_channels,a=sp_rate):
        super().__init__()
        self.resconv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.channel1 = round(out_channels*a)
        self.channel2 = round(out_channels*(a**2))
        self.channel3 = round(out_channels * (a ** 3))
        # self.sort_arg = nn.Parameter(torch.tensor(range(out_channels)),requires_grad=False)
        # self.sort_arg2 = nn.Parameter(torch.tensor(range(self.channel1)),requires_grad=False)
        # self.gap = nn.AdaptiveAvgPool2d(1)
        # self.sort_tensor = torch.zeros((out_channels))
        # self.sort_tensor2 = torch.zeros((self.channel1))

        self.block1 = nn.Sequential(
            nn.Conv2d(self.channel1, self.channel1, 3, padding=1,groups=self.channel1),
            # nn.Conv2d(self.channel1, self.channel1, 3, padding=1),
            nn.BatchNorm2d(self.channel1),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(self.channel2, self.channel2, 3, padding=1),
            # nn.Conv2d(self.channel2, self.channel2, 3, padding=1),
            nn.BatchNorm2d(self.channel2),
            nn.ReLU()
        )
        self.block3_1 = nn.Sequential(
            nn.Conv2d(self.channel3, self.channel3, 3, padding=1),
            # nn.Conv2d(self.channel2, self.channel2, 3, padding=1),
            nn.BatchNorm2d(self.channel3),
            nn.ReLU()
        )

        self.block3_2 = nn.Sequential(
            nn.Conv2d(self.channel3, self.channel3, 3, padding=1),
            # nn.Conv2d(self.channel2, self.channel2, 3, padding=1),
            nn.BatchNorm2d(self.channel3),
            nn.ReLU()
        )

        # self.block2_fu = nn.Sequential(
        #     nn.Conv2d(self.channel1, self.channel1, 1),
        #     nn.BatchNorm2d(self.channel1),
        #     nn.ReLU()
        # )

        self.block4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        # self.block4 = nn.Sequential(
        #     nn.Conv2d(out_channels, out_channels, 3, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU()
        # )
        # self.strip_att = cubic_attention(out_channels, group=1, dilation=1, kernel=3)
        self.strip_att = cubic_attention_2(self.channel1, kernel=3)

        self.fushion = nn.Sequential(
            nn.Conv2d((out_channels - self.channel1) + self.channel1*2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        res = self.resconv(x)
        B,C,H,W = res.shape
        # device = res.device
        # self.sort_tensor = self.sort_tensor.to(device)
        # self.sort_tensor = self.sort_tensor + torch.mean(self.gap(res).reshape((B, C)),dim=0)
        # res = res[:,self.sort_arg]
        part1,part2 = res[:,:self.channel1],res[:,self.channel1:]
        out1 = self.block1(part1)
        # self.sort_tensor2 = self.sort_tensor2.to(device)
        # self.sort_tensor2 = self.sort_tensor2 + torch.mean(self.gap(out1).reshape(out1.shape[:2]), dim=0)
        # out1 = out1[:, self.sort_arg2]
        out1_1,out1_2 = out1[:,:self.channel2],out1[:,self.channel2:]

        out1_1 = self.block2(out1_1)
        # out2 = self.block2_fu(torch.concat((out1_1,out1_2),dim=1))
        out1_1_1,out1_1_2 = out1_1[:,:self.channel3],out1_1[:,self.channel3:]
        out1_1_1 = self.block3_1(out1_1_1)
        out1_1_2 = self.block3_2(out1_1_2)
        out2 = torch.concat((out1_1_1,out1_1_2,out1_2),dim=1)


        out2_res = torch.concat([out2,part2],dim=1)
        out3 = self.block4(out2_res)

        out3_part1,out3_part2 = out3[:,:self.channel1],out3[:,self.channel1:]
        out4 = self.fushion(torch.cat((self.strip_att(out3_part1),out3_part2),dim=1))
        # out4 = self.fushion(torch.cat((self.strip_att(out3_part1),self.block4(out3_part2)),dim=1))

        #
        # return out4 + out3
        return out4
        # return out3

    # def reset_arg(self):
    #     self.sort_arg = nn.Parameter(torch.argsort(self.sort_tensor),requires_grad=False)
    #     self.sort_arg2 = nn.Parameter(torch.argsort(self.sort_tensor2),requires_grad=False)
    #     self.sort_tensor = self.sort_tensor * 0
    #     self.sort_tensor2 = self.sort_tensor2 * 0

class resmspblock_sp_v2(nn.Module):
    def __init__(self, in_channels, out_channels,a=sp_rate):
        super().__init__()
        self.resconv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.channel1 = round(out_channels*a)
        self.channel2 = round(out_channels*(a**2))
        self.channel3 = round(out_channels * (a ** 3))
        self.channel4 = round(out_channels * (a ** 4))
        # self.sort_arg = nn.Parameter(torch.tensor(range(out_channels)),requires_grad=False)
        # self.sort_arg2 = nn.Parameter(torch.tensor(range(self.channel1)),requires_grad=False)
        # self.gap = nn.AdaptiveAvgPool2d(1)
        # self.sort_tensor = torch.zeros((out_channels))
        # self.sort_tensor2 = torch.zeros((self.channel1))

        self.block1 = nn.Sequential(
            nn.Conv2d(self.channel1, self.channel1, 3, padding=1,groups=self.channel1),
            # nn.Conv2d(self.channel1, self.channel1, 3, padding=1),
            nn.BatchNorm2d(self.channel1),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(self.channel2, self.channel2, 3, padding=1),
            # nn.Conv2d(self.channel2, self.channel2, 3, padding=1),
            nn.BatchNorm2d(self.channel2),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(self.channel3, self.channel3, 3, padding=1),
            # nn.Conv2d(self.channel2, self.channel2, 3, padding=1),
            nn.BatchNorm2d(self.channel3),
            nn.ReLU()
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(self.channel4, self.channel4, 3, padding=1),
            # nn.Conv2d(self.channel2, self.channel2, 3, padding=1),
            nn.BatchNorm2d(self.channel4),
            nn.ReLU()
        )

        # self.block2_fu = nn.Sequential(
        #     nn.Conv2d(self.channel1, self.channel1, 1),
        #     nn.BatchNorm2d(self.channel1),
        #     nn.ReLU()
        # )

        self.block5 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        # self.block4 = nn.Sequential(
        #     nn.Conv2d(out_channels, out_channels, 3, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU()
        # )
        # self.strip_att = cubic_attention(out_channels, group=1, dilation=1, kernel=3)
        self.strip_att = cubic_attention_2(self.channel1, kernel=3)

        self.fushion = nn.Sequential(
            nn.Conv2d((out_channels - self.channel1) + self.channel1*2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        res = self.resconv(x)
        B,C,H,W = res.shape
        # device = res.device
        # self.sort_tensor = self.sort_tensor.to(device)
        # self.sort_tensor = self.sort_tensor + torch.mean(self.gap(res).reshape((B, C)),dim=0)
        # res = res[:,self.sort_arg]
        part1,part2 = res[:,:self.channel1],res[:,self.channel1:]
        out1 = self.block1(part1)
        # self.sort_tensor2 = self.sort_tensor2.to(device)
        # self.sort_tensor2 = self.sort_tensor2 + torch.mean(self.gap(out1).reshape(out1.shape[:2]), dim=0)
        # out1 = out1[:, self.sort_arg2]
        out1_1,out1_2 = out1[:,:self.channel2],out1[:,self.channel2:]

        out2 = self.block2(out1_1)
        # out2 = self.block2_fu(torch.concat((out1_1,out1_2),dim=1))
        out2_1,out2_2 = out2[:,:self.channel3],out2[:,self.channel3:]
        out3 = self.block3(out2_1)
        out3_1,out3_2 = out3[:,:self.channel4],out3[:,self.channel4:]
        out3_1 = self.block4(out3_1)
        part1 = torch.concat((out3_1,out3_2,out2_2,out1_2),dim=1)


        out2_res = torch.concat([part1,part2],dim=1)
        out3 = self.block5(out2_res)

        out3_part1,out3_part2 = out3[:,:self.channel1],out3[:,self.channel1:]
        out4 = self.fushion(torch.cat((self.strip_att(out3_part1),out3_part2),dim=1))
        # out4 = self.fushion(torch.cat((self.strip_att(out3_part1),self.block4(out3_part2)),dim=1))

        #
        # return out4 + out3
        return out4
        # return out3

    # def reset_arg(self):
    #     self.sort_arg = nn.Parameter(torch.argsort(self.sort_tensor),requires_grad=False)
    #     self.sort_arg2 = nn.Parameter(torch.argsort(self.sort_tensor2),requires_grad=False)
    #     self.sort_tensor = self.sort_tensor * 0
    #     self.sort_tensor2 = self.sort_tensor2 * 0

class resmspblock_sp_v3(nn.Module):
    def __init__(self, in_channels, out_channels,a=sp_rate):
        super().__init__()
        self.resconv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.channel1 = round((out_channels-out_channels%16)*a)
        self.channel2 = round((out_channels-out_channels%16)*(a**2))
        # self.sort_arg = nn.Parameter(torch.tensor(range(out_channels)),requires_grad=False)
        # self.sort_arg2 = nn.Parameter(torch.tensor(range(self.channel1)),requires_grad=False)
        # self.gap = nn.AdaptiveAvgPool2d(1)
        # self.sort_tensor = torch.zeros((out_channels))
        # self.sort_tensor2 = torch.zeros((self.channel1))

        self.block1 = nn.Sequential(
            nn.Conv2d(self.channel1, self.channel1, 3, padding=1,groups=self.channel1),
            # nn.Conv2d(self.channel1, self.channel1, 3, padding=1),
            nn.BatchNorm2d(self.channel1),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(self.channel2, self.channel2, 3, padding=1,groups=4),
            # nn.Conv2d(self.channel2, self.channel2, 3, padding=1),
            nn.BatchNorm2d(self.channel2),
            nn.ReLU()
        )

        # self.block2_fu = nn.Sequential(
        #     nn.Conv2d(self.channel1, self.channel1, 1),
        #     nn.BatchNorm2d(self.channel1),
        #     nn.ReLU()
        # )

        self.block3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        # self.block4 = nn.Sequential(
        #     nn.Conv2d(out_channels, out_channels, 3, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU()
        # )
        # self.strip_att = cubic_attention(out_channels, group=1, dilation=1, kernel=3)
        self.strip_att = cubic_attention_2(self.channel1, kernel=3)

        self.fushion = nn.Sequential(
            nn.Conv2d((out_channels - self.channel1) + self.channel1*2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        res = self.resconv(x)
        B,C,H,W = res.shape
        # device = res.device
        # self.sort_tensor = self.sort_tensor.to(device)
        # self.sort_tensor = self.sort_tensor + torch.mean(self.gap(res).reshape((B, C)),dim=0)
        # res = res[:,self.sort_arg]
        part1,part2 = res[:,:self.channel1],res[:,self.channel1:]
        out1 = self.block1(part1)
        # self.sort_tensor2 = self.sort_tensor2.to(device)
        # self.sort_tensor2 = self.sort_tensor2 + torch.mean(self.gap(out1).reshape(out1.shape[:2]), dim=0)
        # out1 = out1[:, self.sort_arg2]
        out1_1,out1_2 = out1[:,:self.channel2],out1[:,self.channel2:]

        out1_1 = self.block2(out1_1)
        # out2 = self.block2_fu(torch.concat((out1_1,out1_2),dim=1))
        out2 = torch.concat((out1_1,out1_2),dim=1)

        out2_res = torch.concat([part2,out2],dim=1)
        out3 = self.block3(out2_res)

        out3_part1,out3_part2 = out3[:,:self.channel1],out3[:,self.channel1:]
        out4 = self.fushion(torch.cat((self.strip_att(out3_part1),out3_part2),dim=1))
        # out4 = self.fushion(torch.cat((self.strip_att(out3_part1),self.block4(out3_part2)),dim=1))

        #
        # return out4 + out3
        return out4
        # return out3

    # def reset_arg(self):
    #     self.sort_arg = nn.Parameter(torch.argsort(self.sort_tensor),requires_grad=False)
    #     self.sort_arg2 = nn.Parameter(torch.argsort(self.sort_tensor2),requires_grad=False)
    #     self.sort_tensor = self.sort_tensor * 0
    #     self.sort_tensor2 = self.sort_tensor2 * 0
class gNet_block(nn.Module):
    def __init__(self, in_channels, out_channels,a=sp_rate):
        super().__init__()
        self.resconv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.channel1 = round(out_channels*a)
        self.channel2 = round(out_channels*(a**2))
        self.channel3 = round(out_channels * (a ** 3))
        # self.sort_arg = nn.Parameter(torch.tensor(range(out_channels)),requires_grad=False)
        # self.sort_arg2 = nn.Parameter(torch.tensor(range(self.channel1)),requires_grad=False)
        # self.gap = nn.AdaptiveAvgPool2d(1)
        # self.sort_tensor = torch.zeros((out_channels))
        # self.sort_tensor2 = torch.zeros((self.channel1))

        self.block1_1 = nn.Sequential(
            nn.Conv2d(self.channel1, self.channel1, 1),
            # nn.Conv2d(self.channel1, self.channel1, 3, padding=1),
            nn.BatchNorm2d(self.channel1),
            # nn.ReLU()
        )

        self.block1_2 = nn.Sequential(
            nn.Conv2d(self.channel1, self.channel1, 3, padding=1,groups=self.channel1),
            # nn.Conv2d(self.channel2, self.channel2, 3, padding=1),
            nn.BatchNorm2d(self.channel1),
            nn.Sigmoid()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(self.channel1, self.channel1, 1),
            # nn.Conv2d(self.channel2, self.channel2, 3, padding=1),
            nn.BatchNorm2d(self.channel1),
            nn.ReLU()
        )


        self.fushion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        res = self.resconv(x)
        B,C,H,W = res.shape
        # device = res.device
        # self.sort_tensor = self.sort_tensor.to(device)
        # self.sort_tensor = self.sort_tensor + torch.mean(self.gap(res).reshape((B, C)),dim=0)
        # res = res[:,self.sort_arg]
        part1,part2 = res[:,:self.channel1],res[:,self.channel1:]
        out1_1 = self.block1_1(part1)
        out1_2 = self.block1_2(part1)
        # self.sort_tensor2 = self.sort_tensor2.to(device)
        # self.sort_tensor2 = self.sort_tensor2 + torch.mean(self.gap(out1).reshape(out1.shape[:2]), dim=0)
        # out1 = out1[:, self.sort_arg2]
        out1 = torch.mul(out1_1, out1_2)
        out1 = self.block2(out1)


        out = self.fushion(torch.concat((out1,part2),dim=1))
        # out4 = self.fushion(torch.cat((self.strip_att(out3_part1),self.block4(out3_part2)),dim=1))

        #
        # return out4 + out3
        return out
        # return out3

    # def reset_arg(self):
    #     self.sort_arg = nn.Parameter(torch.argsort(self.sort_tensor),requires_grad=False)
    #     self.sort_arg2 = nn.Parameter(torch.argsort(self.sort_tensor2),requires_grad=False)
    #     self.sort_tensor = self.sort_tensor * 0
    #     self.sort_tensor2 = self.sort_tensor2 * 0

class rm_resmspblock_sp(nn.Module):
    def __init__(self, in_channels, out_channels,a=sp_rate):
        super().__init__()
        self.out_channels = out_channels
        self.channel1 = round(out_channels * a)
        self.channel2 = round(out_channels * (a ** 2))
        self.resconv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.sort_arg = nn.Parameter(torch.randn(out_channels), requires_grad=True)
        self.theta = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.sort_arg2 = nn.Parameter(torch.randn(out_channels), requires_grad=True)
        self.theta2 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.sort_arg3 = nn.Parameter(torch.randn(out_channels*3), requires_grad=True)
        self.theta3 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)


        self.block1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1,groups=out_channels),
            # nn.Conv2d(self.channel1, self.channel1, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            # nn.Conv2d(self.channel2, self.channel2, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(out_channels*3, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.strip_att = cubic_attention_2(self.channel1, kernel=3)

        self.fushion = nn.Sequential(
            nn.Conv2d((out_channels - self.channel1) + self.channel1*2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        res = self.resconv(x)
        # B,C,H,W = res.shape
        sort_arg_1_1 = torch.zeros_like(self.sort_arg)
        sort_arg_1_1[torch.abs(self.sort_arg) >= self.theta] = 1
        sort_arg_1_2 = torch.zeros_like(self.sort_arg)
        sort_arg_1_2[torch.abs(self.sort_arg) < self.theta] = 1
        sort_arg_2_1 = torch.zeros_like(self.sort_arg2)
        sort_arg_2_1[(torch.abs(self.sort_arg2) >= self.theta2) & (torch.abs(self.sort_arg) >= self.theta)] = 1
        sort_arg_2_2 = torch.zeros_like(self.sort_arg2)
        sort_arg_2_2[(torch.abs(self.sort_arg2) < self.theta2) & (torch.abs(self.sort_arg) >= self.theta)] = 1
        sort_arg_3 = torch.zeros_like(self.sort_arg3)
        sort_arg_3[(torch.abs(self.sort_arg3) >= self.theta3)] = 1
        # sort_arg = torch.argsort(torch.mean(self.gap(res).reshape((B, C)),dim=0))
        # res = res[:,sort_arg]
        part1 = res * sort_arg_1_1.unsqueeze(-1).unsqueeze(-1)
        part2 = res * sort_arg_1_2.unsqueeze(-1).unsqueeze(-1)
        # part1,part2 = res[:,:self.channel1],res[:,self.channel1:]
        out1 = self.block1(part1)

        # out1_1,out1_2 = out1[:,:self.channel2],out1[:,self.channel2:]
        out1_1 = out1 * sort_arg_2_1.unsqueeze(-1).unsqueeze(-1)
        out1_2 = out1 * sort_arg_2_2.unsqueeze(-1).unsqueeze(-1)
        out1_1 = self.block2(out1_1)
        # out2 = self.block2_fu(torch.concat((out1_1,out1_2),dim=1))
        out2 = torch.concat((out1_1,out1_2),dim=1)

        out2_res = torch.concat([out2,part2],dim=1)
        out2_res = out2_res * sort_arg_3.unsqueeze(-1).unsqueeze(-1)
        out3 = self.block3(out2_res)

        out3_part1,out3_part2 = out3[:,:self.channel1],out3[:,self.channel1:]
        out4 = self.fushion(torch.cat((self.strip_att(out3_part1),out3_part2),dim=1))
        # out4 = self.fushion(torch.cat((self.strip_att(out3_part1),self.block4(out3_part2)),dim=1))

        #
        # return out4 + out3
        return out4
        # return out3

    def export_block(self):
        return rm_resmspblock_sp_output(self)


class rm_resmspblock_sp_output(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.sort_arg = model.sort_arg >= model.theta
        self.sort_arg2 = (model.sort_arg2[self.sort_arg] >= model.theta2)
        self.sort_arg1_2 = (model.sort_arg2 >= model.theta2) & self.sort_arg
        self.sort_arg1_2_not = self.sort_arg & torch.logical_not(self.sort_arg1_2)
        self.channel1_m = model.channel1
        self.channel2_m = model.channel2
        self.channel1 = torch.count_nonzero(self.sort_arg)
        self.channel1_skip = torch.count_nonzero(torch.logical_not(self.sort_arg))
        self.channel2 = torch.count_nonzero(self.sort_arg2)
        self.channel2_skip = torch.count_nonzero(torch.logical_not(self.sort_arg2))
        with torch.no_grad():
            self.resconv = model.resconv
            self.block1_conv = nn.Conv2d(self.channel1, self.channel1, 3, padding=1, groups=self.channel1)
            self.block1_conv.weight.copy_(model.block1[0].weight[self.sort_arg])
            # self.block1_conv.bias = nn.Parameter(model.block1[0].bias.clone()[self.sort_arg])
            # self.block1_bn = nn.BatchNorm2d(self.channel1)
            # self.block1_bn.running_mean = model.block1[1].running_mean[self.sort_arg]
            # self.block1_bn.running_var = model.block1[1].running_var[self.sort_arg]
            self.block1 = nn.Sequential(
                self.block1_conv,
                # self.block1_bn,
                nn.ReLU()
            )
            print(self.channel1 / model.out_channels, self.channel2 / model.channel1)

            self.block2_conv = nn.Conv2d(self.channel2, model.out_channels, 3, padding=1)
            self.block2_conv.weight.copy_(model.block2[0].weight[:, self.sort_arg1_2])
            # # self.block2_conv.bias = nn.Parameter(model.block2[0].bias.clone())
            # self.block2_bn = nn.BatchNorm2d(model.out_channels)
            # self.block2_bn.running_mean = model.block2[1].running_mean
            # self.block2_bn.running_var = model.block2[1].running_var
            self.block2 = nn.Sequential(
                self.block2_conv,
                # self.block2_bn,
                nn.ReLU()
            )

            all_true = torch.full_like(self.sort_arg, True)
            self.block3_conv = nn.Conv2d(model.out_channels + self.channel2_skip + self.channel1_skip, model.out_channels,
                                         1)
            self.block3_conv.weight.copy_(model.block3[0].weight[:, torch.concat((all_true, self.sort_arg1_2_not, torch.logical_not(self.sort_arg)))])
            # self.block3_conv.bias = nn.Parameter(model.block3[0].bias.clone())
            # self.block3_bn = nn.BatchNorm2d(model.out_channels)
            # self.block3_bn.running_mean = model.block3[1].running_mean
            # self.block3_bn.running_var = model.block3[1].running_var

            self.block3 = nn.Sequential(
                self.block3_conv,
                # self.block3_bn,
                nn.ReLU()
            )

            self.strip_att = model.strip_att

            self.fushion = model.fushion
    def forward(self, x):
        res = self.resconv(x)
        # B,C,H,W = res.shape
        part1 = res[:, self.sort_arg]
        part2 = res[:, torch.logical_not(self.sort_arg)]
        # sort_arg = torch.argsort(torch.mean(self.gap(res).reshape((B, C)),dim=0))
        # res = res[:,sort_arg]
        # part1,part2 = res[:,:self.channel1],res[:,self.channel1:]
        out1 = self.block1(part1)

        # out1_1,out1_2 = out1[:,:self.channel2],out1[:,self.channel2:]
        out1_1 = out1[:, self.sort_arg2]
        out1_2 = out1[:, torch.logical_not(self.sort_arg2)]
        out1_1 = self.block2(out1_1)
        # out2 = self.block2_fu(torch.concat((out1_1,out1_2),dim=1))
        out2 = torch.concat((out1_1,out1_2),dim=1)
        out2_res = torch.concat([out2,part2],dim=1)
        out3 = self.block3(out2_res)

        out3_part1,out3_part2 = out3[:,:self.channel1_m],out3[:,self.channel1_m:]
        out4 = self.fushion(torch.cat((self.strip_att(out3_part1),out3_part2),dim=1))
        # out4 = self.fushion(torch.cat((self.strip_att(out3_part1),self.block4(out3_part2)),dim=1))

        #
        # return out4 + out3
        return out4
        # return out3


class Interpolate(nn.Module):
    def __init__(self, scale_factor = 0.5, mode = "bilinear"):
        super(Interpolate, self).__init__()
        self.interp = F.interpolate
        self.scale_factor = float(scale_factor)
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x
class FSnet(nn.Module):
    def __init__(self,input_channel=3,down_sample = 0,block = resmspblock_sp):
        super().__init__()
        self.level1_block1 = block(input_channel, 64)

        self.level1_block2 = block(64, 64)

        self.level1_block3 = block(96, 96)

        self.level1_block4 = block(150, 150)

        self.level2_block2 = block(64*4 if down_sample==1 else 64, 128)

        self.level2_block3 = block(96, 96)

        self.level2_block4 = block(150, 150)

        self.level3_block3 = block(128*4 if down_sample==1 else 128, 258)

        self.level3_block4 = block(150, 150)

        self.level4_block4 = block(258*4 if down_sample==1 else 258, 512)

        self.out_channel = [150,150,150,512]

        if down_sample==1:
            self.down1 = self.down_sample_wl(64)
            self.down2 = self.down_sample_wl(128)
            self.down3 = self.down_sample_wl(258)
        elif down_sample==0:
            self.down1 = self.down_sample(64, 64)
            # self.down1 = nn.MaxPool2d(2)
            self.down2 = self.down_sample(128, 128)
            # self.down2 = nn.MaxPool2d(2)
            self.down3 = self.down_sample(258, 258)
            # self.down3 = nn.MaxPool2d(2)
        elif down_sample==2:
            self.down1 = Interpolate()
            self.down2 = Interpolate()
            self.down3 = Interpolate()
        elif down_sample == 3:
            self.down1 = nn.MaxPool2d(2)
            self.down2 = nn.MaxPool2d(2)
            self.down3 = nn.MaxPool2d(2)

    def down_sample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def down_sample_wl(self, in_channels):
        return nn.Sequential(
            wt_m(),
            nn.BatchNorm2d(in_channels*4),
        )
    def shuffle(self, levels,mode = 'bilinear'):
        level_num = len(levels)
        out_levels =[]
        for i in range(level_num):
            channels = levels[i].shape[1] // level_num
            out_level = []
            for j in range(level_num):
                l = levels[i][:, channels * j:channels * (j + 1), :, :]
                out_level.append(F.interpolate(l, scale_factor=float(1/(2**(j-i))), mode=mode))
            out_levels.append(out_level)
        output = []
        for i in range(level_num):
            output.append(torch.cat([out_levels[j][i] for j in range(level_num)], dim=1))
        return output

    def forward(self, x):
        # start_time = time.time()
        out1_1 = self.level1_block1(x)
        out1_2 = self.level1_block2(out1_1)
        out1d2 = self.down1(out1_1)
        # end_time = time.time()
        # print(end_time-start_time)

        # start_time = time.time()
        out2_2 = self.level2_block2(out1d2)
        out2d3 = self.down2(out2_2)
        out1_3in, out2_3in = self.shuffle([out1_2, out2_2])
        # end_time = time.time()
        # print(end_time - start_time)

        # start_time = time.time()
        out1_3 = self.level1_block3(out1_3in)
        out2_3 = self.level2_block3(out2_3in)
        out3_3 = self.level3_block3(out2d3)
        out3d4 = self.down3(out3_3)
        # end_time = time.time()
        # print(end_time - start_time)

        # start_time = time.time()
        out1_4in, out2_4in, out3_4in = self.shuffle([out1_3, out2_3, out3_3])
        # end_time = time.time()
        # print(end_time - start_time)

        # start_time = time.time()
        out1 = self.level1_block4(out1_4in)
        out2 = self.level2_block4(out2_4in)
        out3 = self.level3_block4(out3_4in)
        out4 = self.level4_block4(out3d4)
        # end_time = time.time()
        # print(end_time - start_time)

        return out1, out2, out3, out4

class FSnet_s4(nn.Module):
    def __init__(self, input_channel=3,inner_channels = [64,128,258,512],a = 0.5, down_sample=0, block=resblock,model = None):
        super().__init__()
        self.input_channel = input_channel
        self.inner_channels = inner_channels
        self.a = 0.5
        # self.out_channel = [round((inner_channels[0]//3 +inner_channels[1]//3 + inner_channels[2]//3)*a),
        #                      round((inner_channels[0]//3 +inner_channels[1]//3 + inner_channels[2]//3)*a),
        #                      round((inner_channels[0]//3 +inner_channels[1]//3 + inner_channels[2]//3)*a),
        #                      round(inner_channels[3])]
        if model is None:
            self.out_channel = [round((inner_channels[0] // 4 + inner_channels[1] // 4 + inner_channels[2] // 4 +
                                       inner_channels[3] // 4) * a),
                                round((inner_channels[0] // 4 + inner_channels[1] // 4 + inner_channels[2] // 4 +
                                       inner_channels[3] // 4) * a),
                                round((inner_channels[0] // 4 + inner_channels[1] // 4 + inner_channels[2] // 4 +
                                       inner_channels[3] // 4) * a),
                                round((inner_channels[0] // 4 + inner_channels[1] // 4 + inner_channels[2] // 4 +
                                       inner_channels[3] // 4) * a)]
            self.input_conv = block(input_channel, inner_channels[0])
            # 3*64 = 192
            self.level1_block1 = block(inner_channels[0], inner_channels[0])
            # 4094
            self.level1_block2 = block((inner_channels[0]//4 +inner_channels[1]//4 + inner_channels[2]//4 + inner_channels[3]//4),round((inner_channels[0]//4 +inner_channels[1]//4 + inner_channels[2]//4 + inner_channels[3]//4)*a))
            # 22500
            self.level2_block1 = block(inner_channels[0] * 4 if down_sample == 1 else inner_channels[0], inner_channels[1])
            # 8192
            self.level2_block2 = block((inner_channels[0]//4 +inner_channels[1]//4 + inner_channels[2]//4 + inner_channels[3]//4),round((inner_channels[0]//4 +inner_channels[1]//4 + inner_channels[2]//4 + inner_channels[3]//4)*a))
            # 9216
            self.level3_block1 = block(inner_channels[1] * 4 if down_sample == 1 else inner_channels[1], 258)
            # 32768
            self.level3_block2 = block((inner_channels[0]//4 +inner_channels[1]//4 + inner_channels[2]//4 + inner_channels[3]//4),round((inner_channels[0]//4 +inner_channels[1]//4 + inner_channels[2]//4 + inner_channels[3]//4)*a))
            # 22500
            self.level4_block1 = block(inner_channels[2] * 4 if down_sample == 1 else inner_channels[2], inner_channels[3])
            self.level4_block2 = block((inner_channels[0]//4 +inner_channels[1]//4 + inner_channels[2]//4 + inner_channels[3]//4),round((inner_channels[0]//4 +inner_channels[1]//4 + inner_channels[2]//4 + inner_channels[3]//4)*a))
            # 65536
            if down_sample == 1:
                self.down1 = self.down_sample_wl(inner_channels[0])
                self.down2 = self.down_sample_wl(inner_channels[1])
                self.down3 = self.down_sample_wl(inner_channels[2])
            elif down_sample == 0:
                self.down1 = self.down_sample(inner_channels[0], inner_channels[0])
                # self.down1 = nn.MaxPool2d(2)
                self.down2 = self.down_sample(inner_channels[1], inner_channels[1])
                # self.down2 = nn.MaxPool2d(2)
                self.down3 = self.down_sample(inner_channels[2], inner_channels[2])
                # self.down3 = nn.MaxPool2d(2)
            elif down_sample == 2:
                self.down1 = Interpolate()
                self.down2 = Interpolate()
                self.down3 = Interpolate()
            elif down_sample == 3:
                self.down1 = nn.MaxPool2d(2)
                self.down2 = nn.MaxPool2d(2)
                self.down3 = nn.MaxPool2d(2)
        else:
            self.out_channel = [round((model.inner_channels[0] // 4 + model.inner_channels[1] // 4 + model.inner_channels[2] // 4 +
                                       model.inner_channels[3] // 4) * model.a),
                                round((model.inner_channels[0] // 4 + model.inner_channels[1] // 4 + model.inner_channels[2] // 4 +
                                       model.inner_channels[3] // 4) * model.a),
                                round((model.inner_channels[0] // 4 + model.inner_channels[1] // 4 + model.inner_channels[2] // 4 +
                                       model.inner_channels[3] // 4) * model.a),
                                round((model.inner_channels[0] // 4 + model.inner_channels[1] // 4 + model.inner_channels[2] // 4 +
                                       model.inner_channels[3] // 4) * model.a)]
            self.input_conv = model.input_conv.export_block()
            self.level1_block1 = model.level1_block1.export_block()
            # 4094
            self.level1_block2 = model.level1_block2.export_block()
            # 22500
            self.level2_block1 = model.level2_block1.export_block()
            # 8192
            self.level2_block2 = model.level2_block2.export_block()
            # 9216
            self.level3_block1 = model.level3_block1.export_block()
            self.level3_block2 = model.level3_block2.export_block()
            # 22500
            self.level4_block1  = model.level4_block1.export_block()
            self.level4_block2 = model.level4_block2.export_block()
            self.down1 = model.down1
            self.down2 = model.down2
            self.down3 = model.down3

    # def block(self, in_channels, out_channels):
    #     return nn.Sequential(
    #         resblock(in_channels, out_channels),
    #         resblock(out_channels, out_channels)
    #     )


    def reset_arg(self):
        self.level1_block1.reset_arg()
        self.level1_block2.reset_arg()
        self.level2_block1.reset_arg()
        self.level2_block2.reset_arg()
        self.level3_block1.reset_arg()
        self.level3_block2.reset_arg()
        self.level4_block1.reset_arg()
        self.level4_block2.reset_arg()

    def export_model(self):
        return FSnet_s4(model = self)
    def down_sample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def down_sample_wl(self, in_channels):
        return nn.Sequential(
            wt_m(),
            nn.BatchNorm2d(in_channels * 4),
        )

    # part 代表 k 等二或三等分中的第幾份
    # 用插值法做圖片放大縮小
    def shuffle(self, levels, mode='bilinear'):
        level_num = len(levels)
        out_levels = []
        for i in range(level_num):
            channels = levels[i].shape[1] // level_num
            out_level = []
            for j in range(level_num):
                l = levels[i][:, channels * j:channels * (j + 1), :, :]
                out_level.append(F.interpolate(l, scale_factor=float(1 / (2 ** (j - i))), mode=mode))
            out_levels.append(out_level)
        output = []
        for i in range(level_num):
            output.append(torch.cat([out_levels[j][i] for j in range(level_num)], dim=1))
        return output
    def forward(self, x):
        # start_time = time.time()
        out1_1 = self.input_conv(x)
        out1_2 = self.level1_block1(out1_1)
        out1d2 = self.down1(out1_2)



        out2_2 = self.level2_block1(out1d2)
        out2d3 = self.down2(out2_2)


        out3_3 = self.level3_block1(out2d3)
        out3d4 = self.down3(out3_3)


        # start_time = time.time()
        out4_4 = self.level4_block1(out3d4)
        # end_time = time.time()
        # print(end_time - start_time)
        out1, out2, out3,out4 = self.shuffle([out1_2,out2_2,out3_3,out4_4])
        out1 = self.level1_block2(out1)
        out2 = self.level2_block2(out2)
        out3 = self.level3_block2(out3)
        out4 = self.level4_block2(out4)

        return out1, out2, out3, out4

class FSnet_s22(nn.Module):
    def __init__(self, input_channel=3,inner_channels = [64,128,258,512],a = 0.5, down_sample=0, block=resblock):
        super().__init__()
        # self.out_channel = [round((inner_channels[0]//3 +inner_channels[1]//3 + inner_channels[2]//3)*a),
        #                      round((inner_channels[0]//3 +inner_channels[1]//3 + inner_channels[2]//3)*a),
        #                      round((inner_channels[0]//3 +inner_channels[1]//3 + inner_channels[2]//3)*a),
        #                      round(inner_channels[3])]
        self.out_channel = [round((inner_channels[0] // 2 + inner_channels[1] // 2) * a),
                            round((inner_channels[0] // 2 + inner_channels[1] // 2) * a),
                            round((inner_channels[2] // 2 + inner_channels[3] // 2) * a),
                            round((inner_channels[2] // 2 + inner_channels[3] // 2) * a)]
        self.input_conv = block(input_channel, inner_channels[0])
        # 3*64 = 192
        self.level1_block1 = block(inner_channels[0], inner_channels[0])
        # 4094
        self.level1_block2 = block((inner_channels[0] // 2 + inner_channels[1] // 2),round((inner_channels[0] // 2 + inner_channels[1] // 2) * a))
        # 22500
        self.level2_block1 = block(inner_channels[0] * 4 if down_sample == 1 else inner_channels[0], inner_channels[1])
        # 8192
        self.level2_block2 = block((inner_channels[0] // 2 + inner_channels[1] // 2),round((inner_channels[0] // 2 + inner_channels[1] // 2) * a))
        # 9216
        self.level3_block1 = block(inner_channels[1] * 4 if down_sample == 1 else inner_channels[1], 258)
        # 32768
        self.level3_block2 = block((inner_channels[2] // 2 + inner_channels[3] // 2),round((inner_channels[2] // 2 + inner_channels[3] // 2) * a))
        # 22500
        self.level4_block1 = block(inner_channels[2] * 4 if down_sample == 1 else inner_channels[2], inner_channels[3])
        self.level4_block2 = block((inner_channels[2] // 2 + inner_channels[3] // 2),round((inner_channels[2] // 2 + inner_channels[3] // 2) * a))
        # 65536
        if down_sample == 1:
            self.down1 = self.down_sample_wl(inner_channels[0])
            self.down2 = self.down_sample_wl(inner_channels[1])
            self.down3 = self.down_sample_wl(inner_channels[2])
        elif down_sample == 0:
            self.down1 = self.down_sample(inner_channels[0], inner_channels[0])
            # self.down1 = nn.MaxPool2d(2)
            self.down2 = self.down_sample(inner_channels[1], inner_channels[1])
            # self.down2 = nn.MaxPool2d(2)
            self.down3 = self.down_sample(inner_channels[2], inner_channels[2])
            # self.down3 = nn.MaxPool2d(2)
        elif down_sample == 2:
            self.down1 = Interpolate()
            self.down2 = Interpolate()
            self.down3 = Interpolate()
        elif down_sample == 3:
            self.down1 = nn.MaxPool2d(2)
            self.down2 = nn.MaxPool2d(2)
            self.down3 = nn.MaxPool2d(2)

    # def block(self, in_channels, out_channels):
    #     return nn.Sequential(
    #         resblock(in_channels, out_channels),
    #         resblock(out_channels, out_channels)
    #     )

    def down_sample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def down_sample_wl(self, in_channels):
        return nn.Sequential(
            wt_m(),
            nn.BatchNorm2d(in_channels * 4),
        )

    # part 代表 k 等二或三等分中的第幾份
    # 用插值法做圖片放大縮小
    def shuffle(self, levels, mode='bilinear'):
        level_num = len(levels)
        out_levels = []
        for i in range(level_num):
            channels = levels[i].shape[1] // level_num
            out_level = []
            for j in range(level_num):
                l = levels[i][:, channels * j:channels * (j + 1), :, :]
                out_level.append(F.interpolate(l, scale_factor=float(1 / (2 ** (j - i))), mode=mode))
            out_levels.append(out_level)
        output = []
        for i in range(level_num):
            output.append(torch.cat([out_levels[j][i] for j in range(level_num)], dim=1))
        return output
    def forward(self, x):
        # start_time = time.time()
        out1_1 = self.input_conv(x)
        out1_2 = self.level1_block1(out1_1)
        out1d2 = self.down1(out1_2)



        out2_2 = self.level2_block1(out1d2)
        out2d3 = self.down2(out2_2)


        out3_3 = self.level3_block1(out2d3)
        out3d4 = self.down3(out3_3)


        # start_time = time.time()
        out4_4 = self.level4_block1(out3d4)
        # end_time = time.time()
        # print(end_time - start_time)
        # out1, out2, out3,out4 = self.shuffle([out1_2,out2_2,out3_3,out4_4])
        out1, out2 = self.shuffle([out1_2,out2_2])
        out3, out4 = self.shuffle([out3_3, out4_4])
        out1 = self.level1_block2(out1)
        out2 = self.level2_block2(out2)
        out3 = self.level3_block2(out3)
        out4 = self.level4_block2(out4)

        return out1, out2, out3, out4
class FSnet_s3(nn.Module):
    def __init__(self, input_channel=3,inner_channels = [64,128,258,512],a = 0.5, down_sample=0, block=resblock):
        super().__init__()
        self.out_channel = [round((inner_channels[0]//3 +inner_channels[1]//3 + inner_channels[2]//3)*a),
                             round((inner_channels[0]//3 +inner_channels[1]//3 + inner_channels[2]//3)*a),
                             round((inner_channels[0]//3 +inner_channels[1]//3 + inner_channels[2]//3)*a),
                             round(inner_channels[3])]
        self.input_conv = block(input_channel, inner_channels[0])
        # 3*64 = 192
        self.level1_block1 = block(inner_channels[0], inner_channels[0])
        # 4094
        self.level1_block2 = block((inner_channels[0]//3 +inner_channels[1]//3 + inner_channels[2]//3),round((inner_channels[0]//3 +inner_channels[1]//3 + inner_channels[2]//3)*a))
        # 22500
        self.level2_block1 = block(inner_channels[0] * 4 if down_sample == 1 else inner_channels[0], inner_channels[1])
        # 8192
        self.level2_block2 = block((inner_channels[0]//3 +inner_channels[1]//3 + inner_channels[2]//3),round((inner_channels[0]//3 +inner_channels[1]//3 + inner_channels[2]//3)*a))
        # 9216
        self.level3_block1 = block(inner_channels[1] * 4 if down_sample == 1 else inner_channels[1], 258)
        # 32768
        self.level3_block2 = block((inner_channels[0]//3 +inner_channels[1]//3 + inner_channels[2]//3),round((inner_channels[0]//3 +inner_channels[1]//3 + inner_channels[2]//3)*a))
        # 22500
        self.level4_block1 = block(inner_channels[2] * 4 if down_sample == 1 else inner_channels[2], inner_channels[3])
        # 65536
        if down_sample == 1:
            self.down1 = self.down_sample_wl(inner_channels[0])
            self.down2 = self.down_sample_wl(inner_channels[1])
            self.down3 = self.down_sample_wl(inner_channels[2])
        elif down_sample == 0:
            self.down1 = self.down_sample(inner_channels[0], inner_channels[0])
            # self.down1 = nn.MaxPool2d(2)
            self.down2 = self.down_sample(inner_channels[1], inner_channels[1])
            # self.down2 = nn.MaxPool2d(2)
            self.down3 = self.down_sample(inner_channels[2], inner_channels[2])
            # self.down3 = nn.MaxPool2d(2)
        elif down_sample == 2:
            self.down1 = Interpolate()
            self.down2 = Interpolate()
            self.down3 = Interpolate()
        elif down_sample == 3:
            self.down1 = nn.MaxPool2d(2)
            self.down2 = nn.MaxPool2d(2)
            self.down3 = nn.MaxPool2d(2)

    # def block(self, in_channels, out_channels):
    #     return nn.Sequential(
    #         resblock(in_channels, out_channels),
    #         resblock(out_channels, out_channels)
    #     )

    def reset_arg(self):
        self.level1_block1.reset_arg()
        self.level1_block2.reset_arg()
        self.level2_block1.reset_arg()
        self.level2_block2.reset_arg()
        self.level3_block1.reset_arg()
        self.level3_block2.reset_arg()
        self.level4_block1.reset_arg()
    def down_sample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def down_sample_wl(self, in_channels):
        return nn.Sequential(
            wt_m(),
            nn.BatchNorm2d(in_channels * 4),
        )

    # part 代表 k 等二或三等分中的第幾份
    # 用插值法做圖片放大縮小
    def shuffle(self, levels, mode='bilinear'):
        level_num = len(levels)
        out_levels = []
        for i in range(level_num):
            channels = levels[i].shape[1] // level_num
            out_level = []
            for j in range(level_num):
                l = levels[i][:, channels * j:channels * (j + 1), :, :]
                out_level.append(F.interpolate(l, scale_factor=float(1 / (2 ** (j - i))), mode=mode))
            out_levels.append(out_level)
        output = []
        for i in range(level_num):
            output.append(torch.cat([out_levels[j][i] for j in range(level_num)], dim=1))
        return output
    def forward(self, x):
        # start_time = time.time()
        out1_1 = self.input_conv(x)
        out1_2 = self.level1_block1(out1_1)
        out1d2 = self.down1(out1_2)



        out2_2 = self.level2_block1(out1d2)
        out2d3 = self.down2(out2_2)


        out3_3 = self.level3_block1(out2d3)
        out3d4 = self.down3(out3_3)


        # start_time = time.time()
        out4_4 = self.level4_block1(out3d4)
        # end_time = time.time()
        # print(end_time - start_time)
        out1, out2, out3 = self.shuffle([out1_2,out2_2,out3_3])
        out1 = self.level1_block2(out1)
        out2 = self.level2_block2(out2)
        out3 = self.level3_block2(out3)

        return out1, out2, out3, out4_4

class FSnet_P_scale(nn.Module):
    def __init__(self,input_channel=3,block = resblock,xmf = wt_m(requires_grad=False)):
        super().__init__()

        self.xfm = xmf

        self.level1_block1 = block(input_channel, 64)

        self.level1_block2 = block(96, 96)

        self.level1_block3 = block(150, 150)

        self.level2_block1 = block(input_channel, 128)

        self.level2_block2 = block(96, 96)

        self.level2_block3 = block(150, 150)

        self.level3_block1 = block(input_channel, 258)

        self.level3_block2 = block(150, 150)

        self.level4_block1 = block(input_channel, 512)

        self.out_channel = [150,150,150,512]

    def shuffle(self, levels,mode = 'bilinear'):
        level_num = len(levels)
        out_levels =[]
        for i in range(level_num):
            channels = levels[i].shape[1] // level_num
            out_level = []
            for j in range(level_num):
                l = levels[i][:, channels * j:channels * (j + 1), :, :]
                out_level.append(F.interpolate(l, scale_factor=float(1/(2**(j-i))), mode=mode))
            out_levels.append(out_level)
        output = []
        for i in range(level_num):
            output.append(torch.cat([out_levels[j][i] for j in range(level_num)], dim=1))
        return output

    def forward(self, x):
        scale_1 = self.xfm(x[:,[0,4,8]])
        scale_2 = self.xfm(scale_1[:,[0,4,8]])
        scale_3 = self.xfm(scale_2[:,[0,4,8]])
        # start_time = time.time()
        out1_1 = self.level1_block1(x)
        out2_1 = self.level2_block1(scale_1)
        out3_1 = self.level3_block1(scale_2)
        out4_1 = self.level4_block1(scale_3)

        out1_2,out2_2=self.shuffle([out1_1, out2_1])
        out1_2 = self.level1_block2(out1_2)
        out2_2 = self.level2_block2(out2_2)

        out1_3,out2_3,out3_3 = self.shuffle([out1_2, out2_2,out3_1])
        out1_3 = self.level1_block3(out1_3)
        out2_3 = self.level2_block3(out2_3)
        out3_3 = self.level3_block2(out3_3)

        return out1_3, out2_3, out3_3, out4_1

class FSnet_P_scale_S3(nn.Module):
    def __init__(self,input_channel=3,inner_channels = [64,128,258,512],a=0.5,xmf = wt_m(requires_grad=False),block = resblock):
        super().__init__()

        self.xfm = xmf

        self.out_channel = [round((inner_channels[0] // 3 + inner_channels[1] // 3 + inner_channels[2] // 3) * a),
                            round((inner_channels[0] // 3 + inner_channels[1] // 3 + inner_channels[2] // 3) * a),
                            round((inner_channels[0] // 3 + inner_channels[1] // 3 + inner_channels[2] // 3) * a),
                            round(inner_channels[3])]


        self.level1_block1 = block(input_channel, inner_channels[0])

        self.level1_block2 = block((inner_channels[0] // 3 + inner_channels[1] // 3 + inner_channels[2] // 3),round((inner_channels[0] // 3 + inner_channels[1] // 3 + inner_channels[2] // 3)*a))

        self.level2_block1 = block(input_channel, inner_channels[1])

        self.level2_block2 = block((inner_channels[0] // 3 + inner_channels[1] // 3 + inner_channels[2] // 3),round((inner_channels[0] // 3 + inner_channels[1] // 3 + inner_channels[2] // 3)*a))


        self.level3_block1 = block(input_channel, inner_channels[2])

        self.level3_block2 = block((inner_channels[0] // 3 + inner_channels[1] // 3 + inner_channels[2] // 3),round((inner_channels[0] // 3 + inner_channels[1] // 3 + inner_channels[2] // 3)*a))

        self.level4_block1 = block(input_channel, inner_channels[3])
    def shuffle(self, levels,mode = 'bilinear'):
        level_num = len(levels)
        out_levels =[]
        for i in range(level_num):
            channels = levels[i].shape[1] // level_num
            out_level = []
            for j in range(level_num):
                l = levels[i][:, channels * j:channels * (j + 1), :, :]
                out_level.append(F.interpolate(l, scale_factor=float(1/(2**(j-i))), mode=mode))
            out_levels.append(out_level)
        output = []
        for i in range(level_num):
            output.append(torch.cat([out_levels[j][i] for j in range(level_num)], dim=1))
        return output

    def forward(self, x):
        scale_1 = self.xfm(x[:,[0,4,8]])
        scale_2 = self.xfm(scale_1[:,[0,4,8]])
        scale_3 = self.xfm(scale_2[:,[0,4,8]])
        # start_time = time.time()
        out1_1 = self.level1_block1(x)
        out2_1 = self.level2_block1(scale_1)
        out3_1 = self.level3_block1(scale_2)
        out4_1 = self.level4_block1(scale_3)

        out1_2,out2_2,out3_2 = self.shuffle([out1_1, out2_1,out3_1])
        out1_2 = self.level1_block2(out1_2)
        out2_2 = self.level2_block2(out2_2)
        out3_2 = self.level3_block2(out3_2)

        return out1_2, out2_2, out3_2, out4_1

class FSnet_P_scale_S4(nn.Module):
    def __init__(self,input_channel=3,inner_channels = [64,128,258,512],a=0.5,xmf = wt_m(requires_grad=False),block = resblock):
        super().__init__()

        self.xfm = xmf

        self.out_channel = [round(
            (inner_channels[0] // 4 + inner_channels[1] // 4 + inner_channels[2] // 4 + inner_channels[3] // 4) * a),
                            round((inner_channels[0] // 4 + inner_channels[1] // 4 + inner_channels[2] // 4 +
                                   inner_channels[3] // 4) * a),
                            round((inner_channels[0] // 4 + inner_channels[1] // 4 + inner_channels[2] // 4 +
                                   inner_channels[3] // 4) * a),
                            round((inner_channels[0] // 4 + inner_channels[1] // 4 + inner_channels[2] // 4 +
                                   inner_channels[3] // 4) * a)]


        self.level1_block1 = block(input_channel, inner_channels[0])

        self.level1_block2 = block((inner_channels[0]//4 +inner_channels[1]//4 + inner_channels[2]//4 + inner_channels[3]//4),round((inner_channels[0]//4 +inner_channels[1]//4 + inner_channels[2]//4 + inner_channels[3]//4)*a))

        self.level2_block1 = block(input_channel, inner_channels[1])

        self.level2_block2 = block((inner_channels[0]//4 +inner_channels[1]//4 + inner_channels[2]//4 + inner_channels[3]//4),round((inner_channels[0]//4 +inner_channels[1]//4 + inner_channels[2]//4 + inner_channels[3]//4)*a))


        self.level3_block1 = block(input_channel, inner_channels[2])

        self.level3_block2 = block((inner_channels[0]//4 +inner_channels[1]//4 + inner_channels[2]//4 + inner_channels[3]//4),round((inner_channels[0]//4 +inner_channels[1]//4 + inner_channels[2]//4 + inner_channels[3]//4)*a))

        self.level4_block1 = block(input_channel, inner_channels[3])
        self.level4_block2 = block((inner_channels[0]//4 +inner_channels[1]//4 + inner_channels[2]//4 + inner_channels[3]//4),round((inner_channels[0]//4 +inner_channels[1]//4 + inner_channels[2]//4 + inner_channels[3]//4)*a))

    def shuffle(self, levels,mode = 'bilinear'):
        level_num = len(levels)
        out_levels =[]
        for i in range(level_num):
            channels = levels[i].shape[1] // level_num
            out_level = []
            for j in range(level_num):
                l = levels[i][:, channels * j:channels * (j + 1), :, :]
                out_level.append(F.interpolate(l, scale_factor=float(1/(2**(j-i))), mode=mode))
            out_levels.append(out_level)
        output = []
        for i in range(level_num):
            output.append(torch.cat([out_levels[j][i] for j in range(level_num)], dim=1))
        return output

    def forward(self, x):
        scale_1 = self.xfm(x[:,[0,4,8]])
        scale_2 = self.xfm(scale_1[:,[0,4,8]])
        scale_3 = self.xfm(scale_2[:,[0,4,8]])
        # start_time = time.time()
        out1_1 = self.level1_block1(x)
        out2_1 = self.level2_block1(scale_1)
        out3_1 = self.level3_block1(scale_2)
        out4_1 = self.level4_block1(scale_3)

        out1_2,out2_2,out3_2,out4_2 = self.shuffle([out1_1, out2_1,out3_1,out4_1])
        out1_2 = self.level1_block2(out1_2)
        out2_2 = self.level2_block2(out2_2)
        out3_2 = self.level3_block2(out3_2)
        out4_2 = self.level4_block2(out4_2)

        return out1_2, out2_2, out3_2, out4_2

class FSnet_P_scale_S22(nn.Module):
    def __init__(self,input_channel=3,inner_channels = [64,128,258,512],a=0.5,xmf = wt_m(requires_grad=False),block = resmspblock_sp):
        super().__init__()

        self.xfm = xmf

        self.out_channel = [round((inner_channels[0] // 2 + inner_channels[1] // 2) * a),
                            round((inner_channels[0] // 2 + inner_channels[1] // 2) * a),
                            round((inner_channels[2] // 2 + inner_channels[3] // 2) * a),
                            round((inner_channels[2] // 2 + inner_channels[3] // 2) * a)]


        self.level1_block1 = block(input_channel, inner_channels[0])

        self.level1_block2 = block((inner_channels[0] // 2 + inner_channels[1] // 2),
                                     round((inner_channels[0] // 2 + inner_channels[1] // 2) * a))

        self.level2_block1 = block(input_channel, inner_channels[1])

        self.level2_block2 = block((inner_channels[0] // 2 + inner_channels[1] // 2),round((inner_channels[0] // 2 + inner_channels[1] // 2) * a))


        self.level3_block1 = block(input_channel, inner_channels[2])

        self.level3_block2 = block((inner_channels[2] // 2 + inner_channels[3] // 2),round((inner_channels[2] // 2 + inner_channels[3] // 2) * a))

        self.level4_block1 = block(input_channel, inner_channels[3])
        self.level4_block2 = block((inner_channels[2] // 2 + inner_channels[3] // 2),round((inner_channels[2] // 2 + inner_channels[3] // 2) * a))

    def shuffle(self, levels,mode = 'bilinear'):
        level_num = len(levels)
        out_levels =[]
        for i in range(level_num):
            channels = levels[i].shape[1] // level_num
            out_level = []
            for j in range(level_num):
                l = levels[i][:, channels * j:channels * (j + 1), :, :]
                out_level.append(F.interpolate(l, scale_factor=float(1/(2**(j-i))), mode=mode))
            out_levels.append(out_level)
        output = []
        for i in range(level_num):
            output.append(torch.cat([out_levels[j][i] for j in range(level_num)], dim=1))
        return output

    def forward(self, x):
        scale_1 = self.xfm(x[:,[0,4,8]])
        scale_2 = self.xfm(scale_1[:,[0,4,8]])
        scale_3 = self.xfm(scale_2[:,[0,4,8]])
        # start_time = time.time()
        out1_1 = self.level1_block1(x)
        out2_1 = self.level2_block1(scale_1)
        out3_1 = self.level3_block1(scale_2)
        out4_1 = self.level4_block1(scale_3)

        out1_2,out2_2 = self.shuffle([out1_1, out2_1])
        out3_2,out4_2 = self.shuffle([out3_1,out4_1])
        out1_2 = self.level1_block2(out1_2)
        out2_2 = self.level2_block2(out2_2)
        out3_2 = self.level3_block2(out3_2)
        out4_2 = self.level4_block2(out4_2)

        return out1_2, out2_2, out3_2, out4_2


class FSnet_s3_v2(nn.Module):
    def __init__(self, input_channel=3,inner_channels = [64,128,258,512],a = 0.5, down_sample=0,block_1 = mspblock,block_2 = resmspblock_sp):
        super().__init__()
        self.out_channel = [round((inner_channels[0]//3 +inner_channels[1]//3 + inner_channels[2]//3)*a),
                             round((inner_channels[0]//3 +inner_channels[1]//3 + inner_channels[2]//3)*a),
                             round((inner_channels[0]//3 +inner_channels[1]//3 + inner_channels[2]//3)*a),
                             round(inner_channels[3])]
        self.input_conv = block_2(input_channel, inner_channels[0])
        # 3*64 = 192
        self.level1_block1 = block_1(inner_channels[0], inner_channels[0])
        # 4094
        self.level1_block2 = block_2((inner_channels[0]//3 +inner_channels[1]//3 + inner_channels[2]//3),round((inner_channels[0]//3 +inner_channels[1]//3 + inner_channels[2]//3)*a))
        # 22500
        self.level2_block1 = block_1(inner_channels[0] * 4 if down_sample == 1 else inner_channels[0], inner_channels[1])
        # 8192
        self.level2_block2 = block_2((inner_channels[0]//3 +inner_channels[1]//3 + inner_channels[2]//3),round((inner_channels[0]//3 +inner_channels[1]//3 + inner_channels[2]//3)*a))
        # 9216
        self.level3_block1 = block_1(inner_channels[1] * 4 if down_sample == 1 else inner_channels[1], inner_channels[2])
        # 32768
        self.level3_block2 = block_2((inner_channels[0]//3 +inner_channels[1]//3 + inner_channels[2]//3),round((inner_channels[0]//3 +inner_channels[1]//3 + inner_channels[2]//3)*a))
        # 22500
        self.level4_block1 = block_2(inner_channels[2] * 4 if down_sample == 1 else inner_channels[2], inner_channels[3])
        # 65536
        if down_sample == 1:
            self.down1 = self.down_sample_wl(inner_channels[0])
            self.down2 = self.down_sample_wl(inner_channels[1])
            self.down3 = self.down_sample_wl(inner_channels[2])
        elif down_sample == 0:
            self.down1 = self.down_sample(inner_channels[0], inner_channels[0])
            # self.down1 = nn.MaxPool2d(2)
            self.down2 = self.down_sample(inner_channels[1], inner_channels[1])
            # self.down2 = nn.MaxPool2d(2)
            self.down3 = self.down_sample(inner_channels[2], inner_channels[2])
            # self.down3 = nn.MaxPool2d(2)
        elif down_sample == 2:
            self.down1 = Interpolate()
            self.down2 = Interpolate()
            self.down3 = Interpolate()
        elif down_sample == 3:
            self.down1 = nn.MaxPool2d(2)
            self.down2 = nn.MaxPool2d(2)
            self.down3 = nn.MaxPool2d(2)

    # def block(self, in_channels, out_channels):
    #     return nn.Sequential(
    #         resblock(in_channels, out_channels),
    #         resblock(out_channels, out_channels)
    #     )

    def down_sample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def down_sample_wl(self, in_channels):
        return nn.Sequential(
            wt_m(),
            nn.BatchNorm2d(in_channels * 4),
        )

    # part 代表 k 等二或三等分中的第幾份
    # 用插值法做圖片放大縮小
    def shuffle(self, levels, mode='bilinear'):
        level_num = len(levels)
        out_levels = []
        for i in range(level_num):
            channels = levels[i].shape[1] // level_num
            out_level = []
            for j in range(level_num):
                l = levels[i][:, channels * j:channels * (j + 1), :, :]
                out_level.append(F.interpolate(l, scale_factor=float(1 / (2 ** (j - i))), mode=mode))
            out_levels.append(out_level)
        output = []
        for i in range(level_num):
            output.append(torch.cat([out_levels[j][i] for j in range(level_num)], dim=1))
        return output
    def forward(self, x):
        # start_time = time.time()
        out1_1 = self.input_conv(x)
        out1_2 = self.level1_block1(out1_1)
        out1d2 = self.down1(out1_2)



        out2_2 = self.level2_block1(out1d2)
        out2d3 = self.down2(out2_2)


        out3_3 = self.level3_block1(out2d3)
        out3d4 = self.down3(out3_3)


        # start_time = time.time()
        out4_4 = self.level4_block1(out3d4)
        # end_time = time.time()
        # print(end_time - start_time)
        out1, out2, out3 = self.shuffle([out1_2,out2_2,out3_3])
        out1 = self.level1_block2(out1)
        out2 = self.level2_block2(out2)
        out3 = self.level3_block2(out3)

        return out1, out2, out3, out4_4

class FSnet_s4_v2(nn.Module):
    def __init__(self, input_channel=3,inner_channels = [64,128,258,512],a = 0.5, down_sample=0,block_1 = mspblock,block_2 = resmspblock_sp):
        super().__init__()
        # self.out_channel = [round((inner_channels[0]//3 +inner_channels[1]//3 + inner_channels[2]//3)*a),
        #                      round((inner_channels[0]//3 +inner_channels[1]//3 + inner_channels[2]//3)*a),
        #                      round((inner_channels[0]//3 +inner_channels[1]//3 + inner_channels[2]//3)*a),
        #                      round(inner_channels[3])]
        self.out_channel = [round((inner_channels[0] // 4 + inner_channels[1] // 4 + inner_channels[2] // 4 + inner_channels[3] // 4) * a),
                            round((inner_channels[0] // 4 + inner_channels[1] // 4 + inner_channels[2] // 4 + inner_channels[3] // 4) * a),
                            round((inner_channels[0] // 4 + inner_channels[1] // 4 + inner_channels[2] // 4 + inner_channels[3] // 4) * a),
                            round((inner_channels[0] // 4 + inner_channels[1] // 4 + inner_channels[2] // 4 + inner_channels[3] // 4) * a)]
        self.input_conv = block_2(input_channel, inner_channels[0])
        # 3*64 = 192
        self.level1_block1 = block_1(inner_channels[0], inner_channels[0])
        # 4094
        self.level1_block2 = block_2((inner_channels[0]//4 +inner_channels[1]//4 + inner_channels[2]//4 + inner_channels[3]//4),round((inner_channels[0]//4 +inner_channels[1]//4 + inner_channels[2]//4 + inner_channels[3]//4)*a))
        # 22500
        self.level2_block1 = block_1(inner_channels[0] * 4 if down_sample == 1 else inner_channels[0], inner_channels[1])
        # 8192
        self.level2_block2 = block_2((inner_channels[0]//4 +inner_channels[1]//4 + inner_channels[2]//4 + inner_channels[3]//4),round((inner_channels[0]//4 +inner_channels[1]//4 + inner_channels[2]//4 + inner_channels[3]//4)*a))
        # 9216
        self.level3_block1 = block_1(inner_channels[1] * 4 if down_sample == 1 else inner_channels[1], 258)
        # 32768
        self.level3_block2 = block_2((inner_channels[0]//4 +inner_channels[1]//4 + inner_channels[2]//4 + inner_channels[3]//4),round((inner_channels[0]//4 +inner_channels[1]//4 + inner_channels[2]//4 + inner_channels[3]//4)*a))
        # 22500
        self.level4_block1 = block_1(inner_channels[2] * 4 if down_sample == 1 else inner_channels[2], inner_channels[3])
        self.level4_block2 = block_2((inner_channels[0]//4 +inner_channels[1]//4 + inner_channels[2]//4 + inner_channels[3]//4),round((inner_channels[0]//4 +inner_channels[1]//4 + inner_channels[2]//4 + inner_channels[3]//4)*a))
        self.shuffle4 = shuffer_v2(4)
        # 65536
        if down_sample == 1:
            self.down1 = self.down_sample_wl(inner_channels[0])
            self.down2 = self.down_sample_wl(inner_channels[1])
            self.down3 = self.down_sample_wl(inner_channels[2])
        elif down_sample == 0:
            self.down1 = self.down_sample(inner_channels[0], inner_channels[0])
            # self.down1 = nn.MaxPool2d(2)
            self.down2 = self.down_sample(inner_channels[1], inner_channels[1])
            # self.down2 = nn.MaxPool2d(2)
            self.down3 = self.down_sample(inner_channels[2], inner_channels[2])
            # self.down3 = nn.MaxPool2d(2)
        elif down_sample == 2:
            self.down1 = Interpolate()
            self.down2 = Interpolate()
            self.down3 = Interpolate()
        elif down_sample == 3:
            self.down1 = nn.MaxPool2d(2)
            self.down2 = nn.MaxPool2d(2)
            self.down3 = nn.MaxPool2d(2)

    # def block(self, in_channels, out_channels):
    #     return nn.Sequential(
    #         resblock(in_channels, out_channels),
    #         resblock(out_channels, out_channels)
    #     )

    def down_sample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def down_sample_wl(self, in_channels):
        return nn.Sequential(
            wt_m(),
            nn.BatchNorm2d(in_channels * 4),
        )

    # part 代表 k 等二或三等分中的第幾份
    # 用插值法做圖片放大縮小
    def shuffle(self, levels, mode='bilinear'):
        level_num = len(levels)
        out_levels = []
        for i in range(level_num):
            channels = levels[i].shape[1] // level_num
            out_level = []
            for j in range(level_num):
                l = levels[i][:, channels * j:channels * (j + 1), :, :]
                out_level.append(F.interpolate(l, scale_factor=float(1 / (2 ** (j - i))), mode=mode))
            out_levels.append(out_level)
        output = []
        for i in range(level_num):
            output.append(torch.cat([out_levels[j][i] for j in range(level_num)], dim=1))
        return output
    def forward(self, x):
        # start_time = time.time()
        out1_1 = self.input_conv(x)
        out1_2 = self.level1_block1(out1_1)
        out1d2 = self.down1(out1_2)



        out2_2 = self.level2_block1(out1d2)
        out2d3 = self.down2(out2_2)


        out3_3 = self.level3_block1(out2d3)
        out3d4 = self.down3(out3_3)


        # start_time = time.time()
        out4_4 = self.level4_block1(out3d4)
        # end_time = time.time()
        # print(end_time - start_time)
        out1, out2, out3,out4 = self.shuffle([out1_2,out2_2,out3_3,out4_4])
        # out1, out2, out3,out4 = self.shuffle4([out1_2,out2_2,out3_3,out4_4])
        out1 = self.level1_block2(out1)
        out2 = self.level2_block2(out2)
        out3 = self.level3_block2(out3)
        out4 = self.level4_block2(out4)

        return out1, out2, out3, out4

class FSnet_s22_v2(nn.Module):
    def __init__(self, input_channel=3,inner_channels = [64,128,258,512],a = 0.5, down_sample=0,block_1 = mspblock,block_2 = resmspblock_sp):
        super().__init__()
        # self.out_channel = [round((inner_channels[0]//3 +inner_channels[1]//3 + inner_channels[2]//3)*a),
        #                      round((inner_channels[0]//3 +inner_channels[1]//3 + inner_channels[2]//3)*a),
        #                      round((inner_channels[0]//3 +inner_channels[1]//3 + inner_channels[2]//3)*a),
        #                      round(inner_channels[3])]
        self.out_channel = [round((inner_channels[0] // 2 + inner_channels[1] // 2) * a),
                            round((inner_channels[0] // 2 + inner_channels[1] // 2) * a),
                            round((inner_channels[2] // 2 + inner_channels[3] // 2) * a),
                            round((inner_channels[2] // 2 + inner_channels[3] // 2) * a)]
        self.input_conv = block_1(input_channel, inner_channels[0])
        # 3*64 = 192
        self.level1_block1 = block_1(inner_channels[0], inner_channels[0])
        # 4094
        self.level1_block2 = block_2((inner_channels[0] // 2 + inner_channels[1] // 2),round((inner_channels[0] // 2 + inner_channels[1] // 2) * a))
        # 22500
        self.level2_block1 = block_1(inner_channels[0] * 4 if down_sample == 1 else inner_channels[0], inner_channels[1])
        # 8192
        self.level2_block2 = block_2((inner_channels[0] // 2 + inner_channels[1] // 2),round((inner_channels[0] // 2 + inner_channels[1] // 2) * a))
        # 9216
        self.level3_block1 = block_1(inner_channels[1] * 4 if down_sample == 1 else inner_channels[1], 258)
        # 32768
        self.level3_block2 = block_2((inner_channels[2] // 2 + inner_channels[3] // 2),round((inner_channels[2] // 2 + inner_channels[3] // 2) * a))
        # 22500
        self.level4_block1 = block_1(inner_channels[2] * 4 if down_sample == 1 else inner_channels[2], inner_channels[3])
        self.level4_block2 = block_2((inner_channels[2] // 2 + inner_channels[3] // 2),round((inner_channels[2] // 2 + inner_channels[3] // 2) * a))
        # 65536
        if down_sample == 1:
            self.down1 = self.down_sample_wl(inner_channels[0])
            self.down2 = self.down_sample_wl(inner_channels[1])
            self.down3 = self.down_sample_wl(inner_channels[2])
        elif down_sample == 0:
            self.down1 = self.down_sample(inner_channels[0], inner_channels[0])
            # self.down1 = nn.MaxPool2d(2)
            self.down2 = self.down_sample(inner_channels[1], inner_channels[1])
            # self.down2 = nn.MaxPool2d(2)
            self.down3 = self.down_sample(inner_channels[2], inner_channels[2])
            # self.down3 = nn.MaxPool2d(2)
        elif down_sample == 2:
            self.down1 = Interpolate()
            self.down2 = Interpolate()
            self.down3 = Interpolate()
        elif down_sample == 3:
            self.down1 = nn.MaxPool2d(2)
            self.down2 = nn.MaxPool2d(2)
            self.down3 = nn.MaxPool2d(2)

    # def block(self, in_channels, out_channels):
    #     return nn.Sequential(
    #         resblock(in_channels, out_channels),
    #         resblock(out_channels, out_channels)
    #     )

    def down_sample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def down_sample_wl(self, in_channels):
        return nn.Sequential(
            wt_m(),
            nn.BatchNorm2d(in_channels * 4),
        )

    # part 代表 k 等二或三等分中的第幾份
    # 用插值法做圖片放大縮小
    def shuffle(self, levels, mode='bilinear'):
        level_num = len(levels)
        out_levels = []
        for i in range(level_num):
            channels = levels[i].shape[1] // level_num
            out_level = []
            for j in range(level_num):
                l = levels[i][:, channels * j:channels * (j + 1), :, :]
                out_level.append(F.interpolate(l, scale_factor=float(1 / (2 ** (j - i))), mode=mode))
            out_levels.append(out_level)
        output = []
        for i in range(level_num):
            output.append(torch.cat([out_levels[j][i] for j in range(level_num)], dim=1))
        return output
    def forward(self, x):
        # start_time = time.time()
        out1_1 = self.input_conv(x)
        out1_2 = self.level1_block1(out1_1)
        out1d2 = self.down1(out1_2)



        out2_2 = self.level2_block1(out1d2)
        out2d3 = self.down2(out2_2)


        out3_3 = self.level3_block1(out2d3)
        out3d4 = self.down3(out3_3)


        # start_time = time.time()
        out4_4 = self.level4_block1(out3d4)
        # end_time = time.time()
        # print(end_time - start_time)
        # out1, out2, out3,out4 = self.shuffle([out1_2,out2_2,out3_3,out4_4])
        out1, out2 = self.shuffle([out1_2,out2_2])
        out3, out4 = self.shuffle([out3_3, out4_4])
        out1 = self.level1_block2(out1)
        out2 = self.level2_block2(out2)
        out3 = self.level3_block2(out3)
        out4 = self.level4_block2(out4)

        return out1, out2, out3, out4
class FSnet_P_scale_S3_V2(nn.Module):
    def __init__(self,input_channel=3,inner_channels = [64,128,258,512],a=0.5,xmf = wt_m(requires_grad=False),block_1 = mspblock,block_2 = resmspblock_sp):
        super().__init__()

        self.xfm = xmf

        self.out_channel = [round((inner_channels[0] // 3 + inner_channels[1] // 3 + inner_channels[2] // 3) * a),
                            round((inner_channels[0] // 3 + inner_channels[1] // 3 + inner_channels[2] // 3) * a),
                            round((inner_channels[0] // 3 + inner_channels[1] // 3 + inner_channels[2] // 3) * a),
                            round(inner_channels[3])]


        self.level1_block1 = block_1(input_channel, inner_channels[0])

        self.level1_block2 = block_2((inner_channels[0] // 3 + inner_channels[1] // 3 + inner_channels[2] // 3),round((inner_channels[0] // 3 + inner_channels[1] // 3 + inner_channels[2] // 3)*a))

        self.level2_block1 = block_1(input_channel, inner_channels[1])

        self.level2_block2 = block_2((inner_channels[0] // 3 + inner_channels[1] // 3 + inner_channels[2] // 3),round((inner_channels[0] // 3 + inner_channels[1] // 3 + inner_channels[2] // 3)*a))


        self.level3_block1 = block_1(input_channel, inner_channels[2])

        self.level3_block2 = block_2((inner_channels[0] // 3 + inner_channels[1] // 3 + inner_channels[2] // 3),round((inner_channels[0] // 3 + inner_channels[1] // 3 + inner_channels[2] // 3)*a))

        self.level4_block1 = block_1(input_channel, inner_channels[3])
        # self.shuffle3 = shuffer_v2(3)
    def shuffle(self, levels,mode = 'bilinear'):
        level_num = len(levels)
        out_levels =[]
        for i in range(level_num):
            channels = levels[i].shape[1] // level_num
            out_level = []
            for j in range(level_num):
                l = levels[i][:, channels * j:channels * (j + 1), :, :]
                out_level.append(F.interpolate(l, scale_factor=float(1/(2**(j-i))), mode=mode))
            out_levels.append(out_level)
        output = []
        for i in range(level_num):
            output.append(torch.cat([out_levels[j][i] for j in range(level_num)], dim=1))
        return output

    def forward(self, x):
        scale_1 = self.xfm(x[:,[0,4,8]])
        scale_2 = self.xfm(scale_1[:,[0,4,8]])
        scale_3 = self.xfm(scale_2[:,[0,4,8]])
        # start_time = time.time()
        out1_1 = self.level1_block1(x)
        out2_1 = self.level2_block1(scale_1)
        out3_1 = self.level3_block1(scale_2)
        out4_1 = self.level4_block1(scale_3)

        out1_2,out2_2,out3_2 = self.shuffle([out1_1, out2_1,out3_1])
        # out1_2,out2_2,out3_2 = self.shuffle3([out1_1, out2_1,out3_1])
        out1_2 = self.level1_block2(out1_2)
        out2_2 = self.level2_block2(out2_2)
        out3_2 = self.level3_block2(out3_2)

        return out1_2, out2_2, out3_2, out4_1

class FSnet_P_scale_S4_V2(nn.Module):
    def __init__(self,input_channel=3,inner_channels = [64,128,258,512],a=0.5,xmf = wt_m(requires_grad=False),block_1 = mspblock,block_2 = resmspblock_sp):
        super().__init__()

        self.xfm = xmf

        self.out_channel = [round(
            (inner_channels[0] // 4 + inner_channels[1] // 4 + inner_channels[2] // 4 + inner_channels[3] // 4) * a),
                            round((inner_channels[0] // 4 + inner_channels[1] // 4 + inner_channels[2] // 4 +
                                   inner_channels[3] // 4) * a),
                            round((inner_channels[0] // 4 + inner_channels[1] // 4 + inner_channels[2] // 4 +
                                   inner_channels[3] // 4) * a),
                            round((inner_channels[0] // 4 + inner_channels[1] // 4 + inner_channels[2] // 4 +
                                   inner_channels[3] // 4) * a)]


        self.level1_block1 = block_1(input_channel, inner_channels[0])

        self.level1_block2 = block_2((inner_channels[0]//4 +inner_channels[1]//4 + inner_channels[2]//4 + inner_channels[3]//4),round((inner_channels[0]//4 +inner_channels[1]//4 + inner_channels[2]//4 + inner_channels[3]//4)*a))

        self.level2_block1 = block_1(input_channel, inner_channels[1])

        self.level2_block2 = block_2((inner_channels[0]//4 +inner_channels[1]//4 + inner_channels[2]//4 + inner_channels[3]//4),round((inner_channels[0]//4 +inner_channels[1]//4 + inner_channels[2]//4 + inner_channels[3]//4)*a))


        self.level3_block1 = block_1(input_channel, inner_channels[2])

        self.level3_block2 = block_2((inner_channels[0]//4 +inner_channels[1]//4 + inner_channels[2]//4 + inner_channels[3]//4),round((inner_channels[0]//4 +inner_channels[1]//4 + inner_channels[2]//4 + inner_channels[3]//4)*a))

        self.level4_block1 = block_1(input_channel, inner_channels[3])
        self.level4_block2 = block_2((inner_channels[0]//4 +inner_channels[1]//4 + inner_channels[2]//4 + inner_channels[3]//4),round((inner_channels[0]//4 +inner_channels[1]//4 + inner_channels[2]//4 + inner_channels[3]//4)*a))

    def shuffle(self, levels,mode = 'bilinear'):
        level_num = len(levels)
        out_levels =[]
        for i in range(level_num):
            channels = levels[i].shape[1] // level_num
            out_level = []
            for j in range(level_num):
                l = levels[i][:, channels * j:channels * (j + 1), :, :]
                out_level.append(F.interpolate(l, scale_factor=float(1/(2**(j-i))), mode=mode))
            out_levels.append(out_level)
        output = []
        for i in range(level_num):
            output.append(torch.cat([out_levels[j][i] for j in range(level_num)], dim=1))
        return output

    def forward(self, x):
        scale_1 = self.xfm(x[:,[0,4,8]])
        scale_2 = self.xfm(scale_1[:,[0,4,8]])
        scale_3 = self.xfm(scale_2[:,[0,4,8]])
        # start_time = time.time()
        out1_1 = self.level1_block1(x)
        out2_1 = self.level2_block1(scale_1)
        out3_1 = self.level3_block1(scale_2)
        out4_1 = self.level4_block1(scale_3)

        out1_2,out2_2,out3_2,out4_2 = self.shuffle([out1_1, out2_1,out3_1,out4_1])
        out1_2 = self.level1_block2(out1_2)
        out2_2 = self.level2_block2(out2_2)
        out3_2 = self.level3_block2(out3_2)
        out4_2 = self.level4_block2(out4_2)

        return out1_2, out2_2, out3_2, out4_2

class FSnet_P_scale_S22_V2(nn.Module):
    def __init__(self,input_channel=3,inner_channels = [64,128,258,512],a=0.5,xmf = wt_m(requires_grad=False),block_1 = mspblock,block_2 = resmspblock_sp):
        super().__init__()

        self.xfm = xmf

        self.out_channel = [round((inner_channels[0] // 2 + inner_channels[1] // 2) * a),
                            round((inner_channels[0] // 2 + inner_channels[1] // 2) * a),
                            round((inner_channels[2] // 2 + inner_channels[3] // 2) * a),
                            round((inner_channels[2] // 2 + inner_channels[3] // 2) * a)]


        self.level1_block1 = block_1(input_channel, inner_channels[0])

        self.level1_block2 = block_2((inner_channels[0] // 2 + inner_channels[1] // 2),
                                     round((inner_channels[0] // 2 + inner_channels[1] // 2) * a))

        self.level2_block1 = block_1(input_channel, inner_channels[1])

        self.level2_block2 = block_2((inner_channels[0] // 2 + inner_channels[1] // 2),round((inner_channels[0] // 2 + inner_channels[1] // 2) * a))


        self.level3_block1 = block_1(input_channel, inner_channels[2])

        self.level3_block2 = block_2((inner_channels[2] // 2 + inner_channels[3] // 2),round((inner_channels[2] // 2 + inner_channels[3] // 2) * a))

        self.level4_block1 = block_1(input_channel, inner_channels[3])
        self.level4_block2 = block_2((inner_channels[2] // 2 + inner_channels[3] // 2),round((inner_channels[2] // 2 + inner_channels[3] // 2) * a))

    def shuffle(self, levels,mode = 'bilinear'):
        level_num = len(levels)
        out_levels =[]
        for i in range(level_num):
            channels = levels[i].shape[1] // level_num
            out_level = []
            for j in range(level_num):
                l = levels[i][:, channels * j:channels * (j + 1), :, :]
                out_level.append(F.interpolate(l, scale_factor=float(1/(2**(j-i))), mode=mode))
            out_levels.append(out_level)
        output = []
        for i in range(level_num):
            output.append(torch.cat([out_levels[j][i] for j in range(level_num)], dim=1))
        return output

    def forward(self, x):
        scale_1 = self.xfm(x[:,[0,4,8]])
        scale_2 = self.xfm(scale_1[:,[0,4,8]])
        scale_3 = self.xfm(scale_2[:,[0,4,8]])
        # start_time = time.time()
        out1_1 = self.level1_block1(x)
        out2_1 = self.level2_block1(scale_1)
        out3_1 = self.level3_block1(scale_2)
        out4_1 = self.level4_block1(scale_3)

        out1_2,out2_2 = self.shuffle([out1_1, out2_1])
        out3_2,out4_2 = self.shuffle([out3_1,out4_1])
        out1_2 = self.level1_block2(out1_2)
        out2_2 = self.level2_block2(out2_2)
        out3_2 = self.level3_block2(out3_2)
        out4_2 = self.level4_block2(out4_2)

        return out1_2, out2_2, out3_2, out4_2


class UNet_wavelet(nn.Module):
    def __init__(self, in_channels=12,block = resmspblock_sp,model = None):
        super(UNet_wavelet, self).__init__()
        if model is None:
            self.xfm = wt_m(requires_grad=False)  # Accepts all wave types available to PyWavelets
            # inner_channels = [64, 128, 258, 512]

            # 下採樣層
            # self.encoder = FSnet(in_channels,block = block,down_sample=0)
            # self.encoder = FSnet_s22(in_channels,down_sample=0,block = block)
            # self.encoder = FSnet_s4(in_channels,block =block,down_sample=0)
            # self.encoder = FSnet_P_scale(in_channels,block = block,xmf=self.xfm)
            # self.encoder = FSnet_P_scale_S3(in_channels,block = block,xmf=self.xfm)
            self.encoder = FSnet_P_scale_S3_V2(in_channels,xmf=self.xfm)
            # self.encoder = FSnet_s3_v2(in_channels, down_sample=0)

            # self.encoder = FSnet_Sc(in_channels,wl=True)

            # self.xfm = SWTForward(requires_grad=False) # Accepts all wave types available to PyWavelets
            self.ifm = iwt_m(requires_grad=False)
            # self.ifm = SWTInverse(requires_grad=False)
            self.idwt = True
            self.out_ = False

            # self.ll_decoder = decoder(out_channels=3, in_channels=self.encoder.out_channel,block=block)
            self.ll_decoder = decoder_lite3(out_channels=3, in_channels=self.encoder.out_channel,block=block)
            # self.ll_decoder = decoder_lite2(out_channels=3, in_channels=self.encoder.out_channel,block=block)
            # self.ll_decoder = decoder(out_channels=3,in_channels=[75,75,75,512],block = block)
            # self.detail_decoder = decoder(out_channels=9,out_act=nn.Tanh(),block = block)
            # self.detail_decoder = decoder(out_channels=9,out_act=nn.Identity(),in_channels=self.encoder.out_channel,inner_channels=[16,32,64,128],block = block)
            # self.detail_decoder = decoder(out_channels=9,out_act=nn.Identity(),in_channels=self.encoder.out_channel,inner_channels=[16,32,64,128],block = block)
            self.detail_decoder = decoder_lite3(out_channels=9,out_act=nn.Identity(),in_channels=self.encoder.out_channel,inner_channels=[16,32,64,128],block = block)
            # self.detail_decoder = decoder_lite2(out_channels=9,out_act=nn.Identity(),in_channels=self.encoder.out_channel,inner_channels=[16,32,64,128],block = block)
            # self.detail_decoder = decoder(out_channels=9,in_channels=[75,75,75,512],out_act=nn.Identity(),inner_channels=[16,32,64,128],block = block)
        else:
            self.xfm = model.xfm
            self.encoder = model.encoder.export_model()
            # self.encoder = FSnet_P_scale(in_channels,block = block,xmf=self.xfm)
            # self.encoder = FSnet_P_scale_S3(in_channels,block = block,xmf=self.xfm)
            # self.encoder = FSnet_P_scale_S22_V2(in_channels,xmf=self.xfm)
            # self.encoder = FSnet_s22_v2(in_channels, down_sample=0)

            # self.encoder = FSnet_Sc(in_channels,wl=True)

            # self.xfm = SWTForward(requires_grad=False) # Accepts all wave types available to PyWavelets
            self.ifm = model.ifm
            self.idwt = True
            self.out_ = False

            # self.ll_decoder = decoder(out_channels=3, in_channels=self.encoder.out_channel,block=block)
            self.ll_decoder = model.ll_decoder.export_model()
            # self.ll_decoder = decoder(out_channels=3,in_channels=[75,75,75,512],block = block)
            # self.detail_decoder = decoder(out_channels=9,out_act=nn.Tanh(),block = block)
            # self.detail_decoder = decoder(out_channels=9,out_act=nn.Identity(),in_channels=self.encoder.out_channel,inner_channels=[16,32,64,128],block = block)
            self.detail_decoder = model.detail_decoder.export_model()
            # self.detail_decoder = decoder(out_channels=9,in_channels=[75,75,75,512],out_act=nn.Identity(),inner_channels=[16,32,64,128],block = block)

        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        ll_decoder_params = sum(p.numel() for p in self.ll_decoder.parameters())
        detail_decoder_params = sum(p.numel() for p in self.detail_decoder.parameters())
        print("Encoder params: ", encoder_params,"\nll Decoder params: ", ll_decoder_params,"\nDetail Decoder params:",detail_decoder_params)
        # self.scale1_decode = nn.Sequential(
        #     nn.Conv2d(128, 64, 3, padding="same"),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 32, 3, padding="same"),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 3, 1),
        #     nn.ReLU(),
        # )
        # self.scale2_decode = nn.Sequential(
        #     nn.Conv2d(256, 64, 3, padding="same"),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 32, 3, padding="same"),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 3, 1),
        #     nn.ReLU(),
        # )

        # 中間層

        # 上採樣層
        # self.decoder3 = resblock(150+256, 256)
        # self.decoder2 = resblock(149+128, 128)
        # self.decoder1 = resblock(149+64, 64)
        # # self.out_act = nn.Sigmoid()
        # # self.decoder0 = resblock(3 + 32, 32)
        #
        #
        # self.out_LL = nn.Sequential(
        #     nn.Conv2d(149+64, 32, 1),
        #     nn.BatchNorm2d(32),
        #     nn.ELU(),
        #     nn.Conv2d(32, 3, 1),
        #     nn.ELU()
        #     # nn.ReLU()
        #     # nn.Sigmoid()
        #     # nn.Tanh()
        # )
        # self.out_detail = nn.Sequential(
        #     nn.Conv2d(149 + 64, 32, 1),
        #     nn.BatchNorm2d(32),
        #     nn.ELU(),
        #     nn.Conv2d(32, 9, 1),
        #     # nn.ELU()
        #     nn.Tanh()
        # )
        # # self.out = nn.Sequential(
        # #     nn.Conv2d(32, 32, 1),
        # #     nn.BatchNorm2d(32),
        # #     nn.ELU(),
        # #     nn.Conv2d(32, 3, 1),
        # #     nn.ELU(),
        # # )
        #
        # self.up_conv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        # self.up_conv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # self.up_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # self.up_conv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

    def forward(self, x):
        new_x = self.xfm(x)
        # x_scale1 = F.interpolate(new_x, scale_factor=0.5, mode='bilinear')
        # x_scale2 = F.interpolate(new_x, scale_factor=0.25, mode='bilinear')
        # enc1, enc2, enc3, enc4 = self.encoder(new_x,x_scale1,x_scale2)
        # out_ll,out_ll_scale1,out_ll_scale2 = self.ll_decoder(enc1, enc2, enc3, enc4)
        # out_ll_scale1 = self.scale1_decode(out_ll_scale1)
        # out_ll_scale2 = self.scale2_decode(out_ll_scale2)
        enc1, enc2, enc3, enc4 = self.encoder(new_x)
        out_ll,dec2_out_ll,dec3_out_ll = self.ll_decoder(enc1, enc2, enc3, enc4)
        # out_ll = torch.clamp(out_ll, min=0, max=1)
        out_detail,dec2_out_detail,dec3_out_detail = self.detail_decoder(enc1, enc2, enc3, enc4)
        # out_detail = torch.clamp(out_detail, min=-1, max=1)

        if self.idwt:
            recon_R = self.ifm(torch.cat((out_ll[:, [0]], out_detail[:, 0:3]), dim=1))
            # recon_R = (recon_R - torch.min(recon_R))/(torch.max(recon_R) - torch.min(recon_R))
            recon_R = torch.clamp(recon_R, min=0, max=1)
            # recon_R = self.out_act(recon_R)
            recon_G = self.ifm(torch.cat((out_ll[:, [1]], out_detail[:, 3:6]), dim=1))
            # recon_G = (recon_G - torch.min(recon_G))/(torch.max(recon_G) - torch.min(recon_G))
            # recon_G = self.out_act(recon_G)
            recon_G = torch.clamp(recon_G, min=0, max=1)
            recon_B = self.ifm(torch.cat((out_ll[:, [2]], out_detail[:, 6:9]), dim=1))
            # recon_B = (recon_B - torch.min(recon_B))/(torch.max(recon_B) - torch.min(recon_B))
    #         recon_B = self.out_act(recon_B)
            recon_B = torch.clamp(recon_B, min=0, max=1)
            output_map = torch.cat((recon_R, recon_G, recon_B), dim=1)
            # return (output_map, ((out_ll,out_ll_scale1,out_ll_scale2), out_detail), enc2, enc3, enc4)
            if self.out_:
                return output_map
            else:
                return (output_map, (out_ll, out_detail), torch.concat((dec2_out_ll,dec2_out_detail),dim=1), torch.concat((dec3_out_ll, dec3_out_detail), dim=1))
        else:
            return out_ll, out_detail,torch.concat((dec2_out_ll,dec2_out_detail),dim=1), torch.concat((dec3_out_ll, dec3_out_detail), dim=1)
    def forward_stage_0(self,x):
        return self.xfm(x)

    def forward_stage_1(self, x):
        enc1, enc2, enc3, enc4 = self.encoder(x)
        return enc1, enc2, enc3, enc4
    def forward_stage_2(self, enc1, enc2, enc3, enc4):
        out_ll = self.ll_decoder(enc1, enc2, enc3, enc4)[0]
        return out_ll
    def forward_stage_3(self, enc1, enc2, enc3, enc4):
        out_detail = self.detail_decoder(enc1, enc2, enc3, enc4)[0]
        out_detail = torch.clamp(out_detail, min=-1, max=1)
        return out_detail
    def forward_stage_4(self, out_ll,out_detail):
        recon_R = self.ifm(torch.cat((out_ll[:, [0]], out_detail[:, 0:3]), dim=1))
        # recon_R = (recon_R - torch.min(recon_R))/(torch.max(recon_R) - torch.min(recon_R))
        recon_R = torch.clamp(recon_R, min=-1, max=1)
        # recon_R = self.out_act(recon_R)
        recon_G = self.ifm(torch.cat((out_ll[:, [1]], out_detail[:, 3:6]), dim=1))
        # recon_G = (recon_G - torch.min(recon_G))/(torch.max(recon_G) - torch.min(recon_G))
        # recon_G = self.out_act(recon_G)
        recon_G = torch.clamp(recon_G, min=-1, max=1)
        recon_B = self.ifm(torch.cat((out_ll[:, [2]], out_detail[:, 6:9]), dim=1))
        # recon_B = (recon_B - torch.min(recon_B))/(torch.max(recon_B) - torch.min(recon_B))
        #         recon_B = self.out_act(recon_B)
        recon_B = torch.clamp(recon_B, min=-1, max=1)
        output_map = torch.cat((recon_R, recon_G, recon_B), dim=1)
        return output_map

    def ll_scale(self, x):
        new_x = self.xfm(x)[:,[0,4,8]]
        x_scale1 = F.interpolate(new_x, scale_factor=0.5, mode='bilinear')
        x_scale2 = F.interpolate(new_x, scale_factor=0.25, mode='bilinear')
        return x_scale1, x_scale2

    def reset_arg(self):
        self.encoder.reset_arg()
        self.ll_decoder.reset_arg()
        self.detail_decoder.reset_arg()

    def export_model(self):
        return UNet_wavelet(model=self)