import torch
import torch.nn as nn
import torch.nn.functional as F
from SWT import SWTForward,SWTInverse
from wavelet import wt_m,iwt_m


class UNet_wavelet(nn.Module):
    def __init__(self, in_channels=12, out_channels=12):
        super(UNet_wavelet, self).__init__()

        # 下採樣層
        self.encoder = FSnet(in_channels)
        # self.encoder = FSnet_Sc(in_channels,wl=True)

        self.xfm = wt_m(requires_grad=False) # Accepts all wave types available to PyWavelets
        # self.xfm = SWTForward(requires_grad=False) # Accepts all wave types available to PyWavelets
        self.ifm = iwt_m(requires_grad=False)
        # self.ifm = SWTInverse(requires_grad=False)

        self.ll_decoder = decoder(out_channels=3)
        self.detail_decoder = decoder(out_channels=9,out_act=nn.Tanh())

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

    def forward(self, x, decoder=True):
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
        # out_detail = torch.clamp(out_detail, min=-1, max=1)

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





        # if decoder:
        #     dec4_up = self.up_conv4(enc4)
        #     # dec4_up = self.ifm(enc4)
        #
        #     dec3_skip = enc3
        #     dec3_in = torch.cat((dec4_up, dec3_skip), dim=1)
        #     dec3_out = self.decoder3(dec3_in)
        #     dec3_up = self.up_conv3(dec3_out)
        #     # dec3_up = self.ifm(dec3_out)
        #
        #     dec2_skip = enc2
        #     dec2_in = torch.cat((dec3_up, dec2_skip), dim=1)
        #     dec2_out = self.decoder2(dec2_in)
        #     dec2_up = self.up_conv2(dec2_out)
        #     # dec2_up = self.ifm(dec2_out)
        #
        #     dec1_skip = enc1
        #     dec1_in = torch.cat((dec2_up, dec1_skip), dim=1)
        #     # output1 = self.decoder1(dec1_in)
        #
        #     # dec1_out = self.decoder1(dec1_in)
        #     # dec1_up = self.up_conv1(dec1_out)
        #     #
        #     # dec0_in = torch.cat((dec1_up,x), dim=1)
        #     # dec0_out = self.decoder0(dec0_in)
        #     # output_map = self.out(dec0_out)
        #
        #
        #     # output1 = self.out(dec1_in)
        #
        #     # output_map = self.ifm(output1)
        #     out_ll = self.out_LL(dec1_in)
        #     out_detail = self.out_detail(dec1_in)
        #     recon_R = self.ifm(torch.cat((out_ll[:, [0]], out_detail[:, 0:3]), dim=1))
        #     recon_G = self.ifm(torch.cat((out_ll[:, [1]], out_detail[:, 3:6]), dim=1))
        #     recon_B = self.ifm(torch.cat((out_ll[:, [2]], out_detail[:, 6:9]), dim=1))
        #     output_map = torch.cat((recon_R, recon_G, recon_B), dim=1)
        #     # output_map = self.out(output_map)
        #
        #     # return output1
        #     # return (output_map, enc1, enc2, enc3, enc4)
        #     # return (output_map, output1)
        #     # return (output_map, output1, dec2_out, dec3_out, enc4)
        #     return (output_map, (out_ll,out_detail), dec2_out, dec3_out, enc4)
        #     # return (out_detail,out_ll, dec2_out, dec3_out, enc4)
        # else:
        #     return (enc1, enc2, enc3, enc4)
    def ll_scale(self, x):
        new_x = self.xfm(x)[:,[0,4,8]]
        x_scale1 = F.interpolate(new_x, scale_factor=0.5, mode='bilinear')
        x_scale2 = F.interpolate(new_x, scale_factor=0.25, mode='bilinear')
        return x_scale1, x_scale2


class decoder(nn.Module):
    def __init__(self, in_channels=[150,149,149], out_channels=3,out_act = nn.ELU()):
        super(decoder, self).__init__()
        self.xfm = wt_m(requires_grad=False)  # Accepts all wave types available to PyWavelets
        self.ifm = iwt_m(requires_grad=False)

        self.decoder3 = resblock(in_channels[0] + 256, 256)
        self.decoder2 = resblock(in_channels[1] + 128, 128)
        self.decoder1 = resblock(in_channels[2] + 64, 64)
        # self.decoder0 = resblock(3 + 32, 32)

        self.out = nn.Sequential(
            nn.Conv2d(149 + 64, 32, 1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, out_channels, 1),
            # nn.ELU()
            out_act
        )
        self.up_conv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

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
    def __init__(self,in_channels=3,out_channels=3):
        super(UNet, self).__init__()

        # 下採樣層
        self.encoder = FSnet(in_channels)

        # 中間層
        
        # 上採樣層
        self.decoder3 = resblock(406, 256)
        self.decoder2 = resblock(277, 128)
        self.decoder1 = resblock(213, 64)
        self.out = nn.Sequential(
            nn.Conv2d(64, out_channels, 1),
            nn.Sigmoid()
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

            # return output1
            return (output1,enc1, enc2, enc3, enc4)
        else:
            return (enc1, enc2, enc3, enc4)


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
    

# level 代表哪一層，block哪一個
class FSnet(nn.Module):
    def __init__(self,input_channel=3,wl = False):
        super().__init__()
        self.level1_block1 = resblock(input_channel, 64)

        self.level1_block2 = resblock(64, 64)

        self.level1_block3 = resblock(96, 96)

        self.level1_block4 = resblock(149, 149)

        self.level2_block2 = resblock(64*4 if wl else 64, 128)

        self.level2_block3 = resblock(96, 96)

        self.level2_block4 = resblock(149, 149)

        self.level3_block3 = resblock(128*4 if wl else 128, 256)

        self.level3_block4 = resblock(150, 150)

        self.level4_block4 = resblock(256*4 if wl else 256, 512)

        if wl:
            self.down1 = self.down_sample_wl(64)
            self.down2 = self.down_sample_wl(128)
            self.down3 = self.down_sample_wl(256)
        else:
            self.down1 = self.down_sample(64, 64)
            self.down2 = self.down_sample(128, 128)
            self.down3 = self.down_sample(256, 256)

    def block(self, in_channels, out_channels):
        return nn.Sequential(
            resblock(in_channels, out_channels),
            resblock(out_channels, out_channels)
        )

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

    # part 代表 k 等二或三等分中的第幾份
    # 用插值法做圖片放大縮小
    def shuffle1(self, level1, level2):
        channels = level1.shape[1] // 2
        level1_part1 = level1[:, :channels, :, :]
        level1_part2 = level1[:, channels:, :, :]
        level1_part2 = F.interpolate(level1_part2, scale_factor=0.5, mode='bilinear')
        
        channels = level2.shape[1] // 2
        level2_part1 = level2[:, :channels, :, :]
        level2_part1 = F.interpolate(level2_part1, scale_factor=2, mode='bilinear')
        level2_part2 = level2[:, channels:, :, :]

        level1 = torch.cat((level1_part1, level2_part1), dim=1)
        level2 = torch.cat((level1_part2, level2_part2), dim=1)
        return level1, level2
    
    def shuffle2(self, level1, level2, level3):
        channels = level1.shape[1] // 3
        level1_part1 = level1[:, :channels, :, :]
        level1_part2 = level1[:, channels:2*channels, :, :]
        level1_part2 = F.interpolate(level1_part2, scale_factor=0.5, mode='bilinear')
        level1_part3 = level1[:, 2*channels:, :, :]
        level1_part3 = F.interpolate(level1_part3, scale_factor=0.25, mode='bicubic')
        
        channels = level2.shape[1] // 3
        level2_part1 = level2[:, :channels, :, :]
        level2_part1 = F.interpolate(level2_part1, scale_factor=2, mode='bilinear')
        level2_part2 = level2[:, channels:2*channels, :, :]
        level2_part3 = level2[:, 2*channels:, :, :]
        level2_part3 = F.interpolate(level2_part3, scale_factor=0.5, mode='bilinear')

        channels = level3.shape[1] // 3
        level3_part1 = level3[:, :channels, :, :]
        level3_part1 = F.interpolate(level3_part1, scale_factor=4, mode='bicubic')
        level3_part2 = level3[:, channels:2*channels, :, :]
        level3_part2 = F.interpolate(level3_part2, scale_factor=2, mode='bilinear')
        level3_part3 = level3[:, 2*channels:, :, :]

        level1 = torch.cat((level1_part1, level2_part1, level3_part1), dim=1)
        level2 = torch.cat((level1_part2, level2_part2, level3_part2), dim=1)
        level3 = torch.cat((level1_part3, level2_part3, level3_part3), dim=1)
        return level1, level2, level3

    def forward(self, x):
        out1_1 = self.level1_block1(x)
        out1_2 = self.level1_block2(out1_1)
        out1d2 = self.down1(out1_1)

        out2_2 = self.level2_block2(out1d2)
        out2d3 = self.down2(out2_2)
        out1_3in, out2_3in = self.shuffle1(out1_2, out2_2)

        out1_3 = self.level1_block3(out1_3in)
        out2_3 = self.level2_block3(out2_3in)
        out3_3 = self.level3_block3(out2d3)
        out3d4 = self.down3(out3_3)

        out1_4in, out2_4in, out3_4in = self.shuffle2(out1_3, out2_3, out3_3)

        out1 = self.level1_block4(out1_4in)
        out2 = self.level2_block4(out2_4in)
        out3 = self.level3_block4(out3_4in)
        out4 = self.level4_block4(out3d4)

        return out1, out2, out3, out4


class FSnet_Sc(nn.Module):
    def __init__(self, input_channel=3, wl=False):
        super().__init__()
        self.level1_block1 = resblock(input_channel, 64)

        self.level1_block2 = resblock(64, 64)

        self.level1_block3 = resblock(96, 96)

        self.level1_block4 = resblock(149, 149)

        self.scale1_necode = self.out = nn.Sequential(
            nn.Conv2d(12, 32, 3, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding="same"),
            nn.ReLU(),
        )

        self.level2_block2 = resblock((64 * 4 if wl else 64) + 64, 128)

        self.level2_block3 = resblock(96, 96)

        self.level2_block4 = resblock(149, 149)

        self.scale2_necode = self.out = nn.Sequential(
            nn.Conv2d(12, 32, 3, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding="same"),
            nn.ReLU(),
        )

        self.level3_block3 = resblock((128 * 4 if wl else 128) + 64, 256)

        self.level3_block4 = resblock(150, 150)

        self.level4_block4 = resblock(256 * 4 if wl else 256, 512)

        if wl:
            self.down1 = self.down_sample_wl(64)
            self.down2 = self.down_sample_wl(128)
            self.down3 = self.down_sample_wl(256)
        else:
            self.down1 = self.down_sample(64, 64)
            self.down2 = self.down_sample(128, 128)
            self.down3 = self.down_sample(256, 256)

    def block(self, in_channels, out_channels):
        return nn.Sequential(
            resblock(in_channels, out_channels),
            resblock(out_channels, out_channels)
        )

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
    def shuffle1(self, level1, level2):
        channels = level1.shape[1] // 2
        level1_part1 = level1[:, :channels, :, :]
        level1_part2 = level1[:, channels:, :, :]
        level1_part2 = F.interpolate(level1_part2, scale_factor=0.5, mode='bilinear')

        channels = level2.shape[1] // 2
        level2_part1 = level2[:, :channels, :, :]
        level2_part1 = F.interpolate(level2_part1, scale_factor=2, mode='bilinear')
        level2_part2 = level2[:, channels:, :, :]

        level1 = torch.cat((level1_part1, level2_part1), dim=1)
        level2 = torch.cat((level1_part2, level2_part2), dim=1)
        return level1, level2

    def shuffle2(self, level1, level2, level3):
        channels = level1.shape[1] // 3
        level1_part1 = level1[:, :channels, :, :]
        level1_part2 = level1[:, channels:2 * channels, :, :]
        level1_part2 = F.interpolate(level1_part2, scale_factor=0.5, mode='bilinear')
        level1_part3 = level1[:, 2 * channels:, :, :]
        level1_part3 = F.interpolate(level1_part3, scale_factor=0.25, mode='bicubic')

        channels = level2.shape[1] // 3
        level2_part1 = level2[:, :channels, :, :]
        level2_part1 = F.interpolate(level2_part1, scale_factor=2, mode='bilinear')
        level2_part2 = level2[:, channels:2 * channels, :, :]
        level2_part3 = level2[:, 2 * channels:, :, :]
        level2_part3 = F.interpolate(level2_part3, scale_factor=0.5, mode='bilinear')

        channels = level3.shape[1] // 3
        level3_part1 = level3[:, :channels, :, :]
        level3_part1 = F.interpolate(level3_part1, scale_factor=4, mode='bicubic')
        level3_part2 = level3[:, channels:2 * channels, :, :]
        level3_part2 = F.interpolate(level3_part2, scale_factor=2, mode='bilinear')
        level3_part3 = level3[:, 2 * channels:, :, :]

        level1 = torch.cat((level1_part1, level2_part1, level3_part1), dim=1)
        level2 = torch.cat((level1_part2, level2_part2, level3_part2), dim=1)
        level3 = torch.cat((level1_part3, level2_part3, level3_part3), dim=1)
        return level1, level2, level3

    def forward(self, x,scale_1,scale_2):
        out1_1 = self.level1_block1(x)
        out1_2 = self.level1_block2(out1_1)
        out1d2 = self.down1(out1_1)

        x_s1 = self.scale1_necode(scale_1)
        out1d2 = torch.cat((out1d2,x_s1),dim=1)
        out2_2 = self.level2_block2(out1d2)
        out2d3 = self.down2(out2_2)
        out1_3in, out2_3in = self.shuffle1(out1_2, out2_2)

        x_s2 = self.scale2_necode(scale_2)
        out2d3 = torch.cat((out2d3, x_s2),dim=1)
        out1_3 = self.level1_block3(out1_3in)
        out2_3 = self.level2_block3(out2_3in)
        out3_3 = self.level3_block3(out2d3)
        out3d4 = self.down3(out3_3)

        out1_4in, out2_4in, out3_4in = self.shuffle2(out1_3, out2_3, out3_3)

        out1 = self.level1_block4(out1_4in)
        out2 = self.level2_block4(out2_4in)
        out3 = self.level3_block4(out3_4in)
        out4 = self.level4_block4(out3d4)

        return out1, out2, out3, out4