import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse


from SWT import SWTForward,SWTInverse
from wavelet import wt_m,iwt_m


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

class UNet_wavelet(nn.Module):
    def __init__(self, in_channels=12,block =resblock):
        super(UNet_wavelet, self).__init__()

        self.xfm = wt_m(requires_grad=False)  # Accepts all wave types available to PyWavelets
        # inner_channels = [64, 128, 258, 512]

        # 下採樣層
        self.encoder = FSnet(in_channels,block = block,down_sample=0)
        # self.encoder = FSnet_s3(in_channels,block = block,down_sample=0)
        # self.encoder = FSnet_P_scale(in_channels,block = block,xmf=self.xfm)
        # self.encoder = FSnet_P_scale_S4(in_channels,block = block,xmf=self.xfm)

        # self.encoder = FSnet_Sc(in_channels,wl=True)

        # self.xfm = SWTForward(requires_grad=False) # Accepts all wave types available to PyWavelets
        self.ifm = iwt_m(requires_grad=False)
        # self.ifm = SWTInverse(requires_grad=False)
        self.idwt = True

        self.ll_decoder = decoder(out_channels=3, in_channels=self.encoder.out_channel,block=block)
        # self.ll_decoder = decoder(out_channels=3,in_channels=[75,75,75,512],block = block)
        # self.detail_decoder = decoder(out_channels=9,out_act=nn.Tanh(),block = block)
        self.detail_decoder = decoder(out_channels=9,out_act=nn.Identity(),in_channels=self.encoder.out_channel,inner_channels=[16,32,64,128],block = block)
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
        out_ll = self.ll_decoder(enc1, enc2, enc3, enc4)[0]
        # out_ll = torch.clamp(out_ll, min=0, max=1)
        out_detail = self.detail_decoder(enc1, enc2, enc3, enc4)[0]
        out_detail = torch.clamp(out_detail, min=-1, max=1)

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
            return (output_map, (out_ll, out_detail), enc2, enc3, enc4)
        else:
            return out_ll, out_detail, enc2, enc3, enc4
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
        return output_map

    def ll_scale(self, x):
        new_x = self.xfm(x)[:,[0,4,8]]
        x_scale1 = F.interpolate(new_x, scale_factor=0.5, mode='bilinear')
        x_scale2 = F.interpolate(new_x, scale_factor=0.25, mode='bilinear')
        return x_scale1, x_scale2


class decoder(nn.Module):
    def __init__(self, in_channels=[150,150,150,512],inner_channels = [32,64,128,256], out_channels=3,out_act = nn.ELU(),block = resblock):
        super(decoder, self).__init__()
        self.xfm = wt_m(requires_grad=False)  # Accepts all wave types available to PyWavelets
        self.ifm = iwt_m(requires_grad=False)

        self.decoder3 = block(in_channels[2] + inner_channels[3], inner_channels[3])
        self.decoder2 = block(in_channels[1] + inner_channels[2], inner_channels[2])
        self.decoder1 = block(in_channels[0] + inner_channels[1], inner_channels[1])
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

        return out4 + out2_res

class resmspblock(nn.Module):
    def __init__(self, in_channels, out_channels,a = 0.5):
        super().__init__()
        self.resconv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.channel1 = round(out_channels*a)
        self.channel2 = round(out_channels*(a**2))

        self.block1 = nn.Sequential(
            nn.Conv2d(self.channel1, self.channel1, 3, padding=1),
            nn.BatchNorm2d(self.channel1),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(self.channel2, self.channel2, 3, padding=1,groups=self.channel2),
            # nn.Conv2d(self.channel2, self.channel2, 3, padding=1),
            # nn.Conv2d(out_channels//4, out_channels//4, 1),
            nn.BatchNorm2d(self.channel2),
            nn.ReLU()
        )

        self.block2_fu = nn.Sequential(
            nn.Conv2d(self.channel1, self.channel1, 1),
            nn.BatchNorm2d(self.channel1),
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

        return out4 + out2_res

# level 代表哪一層，block
#
class Interpolate(nn.Module):
    def __init__(self, scale_factor = 0.5, mode = "bilinear"):
        super(Interpolate, self).__init__()
        self.interp = F.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x
class FSnet(nn.Module):
    def __init__(self,input_channel=3,down_sample = 0,block = resblock):
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
                out_level.append(F.interpolate(l, scale_factor=1/(2**(j-i)), mode=mode))
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
                out_level.append(F.interpolate(l, scale_factor=1/(2**(j-i)), mode=mode))
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
                out_level.append(F.interpolate(l, scale_factor=1/(2**(j-i)), mode=mode))
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
                out_level.append(F.interpolate(l, scale_factor=1/(2**(j-i)), mode=mode))
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

class FSnet_s4(nn.Module):
    def __init__(self, input_channel=3,inner_channels = [64,128,258,512],a = 0.5, down_sample=0, block=resblock):
        super().__init__()
        # self.out_channel = [round((inner_channels[0]//3 +inner_channels[1]//3 + inner_channels[2]//3)*a),
        #                      round((inner_channels[0]//3 +inner_channels[1]//3 + inner_channels[2]//3)*a),
        #                      round((inner_channels[0]//3 +inner_channels[1]//3 + inner_channels[2]//3)*a),
        #                      round(inner_channels[3])]
        self.out_channel = [round((inner_channels[0] // 4 + inner_channels[1] // 4 + inner_channels[2] // 4 + inner_channels[3] // 4) * a),
                            round((inner_channels[0] // 4 + inner_channels[1] // 4 + inner_channels[2] // 4 + inner_channels[3] // 4) * a),
                            round((inner_channels[0] // 4 + inner_channels[1] // 4 + inner_channels[2] // 4 + inner_channels[3] // 4) * a),
                            round((inner_channels[0] // 4 + inner_channels[1] // 4 + inner_channels[2] // 4 + inner_channels[3] // 4) * a)]
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
                out_level.append(F.interpolate(l, scale_factor=1 / (2 ** (j - i)), mode=mode))
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
                out_level.append(F.interpolate(l, scale_factor=1 / (2 ** (j - i)), mode=mode))
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