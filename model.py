import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from SWT import SWTForward,SWTInverse


class RB_Block(nn.Module):
    def __init__(self, in_channels, inner_channel = 64):
        super(RB_Block, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, inner_channel,
                                    kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(inner_channel),
            nn.PReLU(),
            nn.Conv2d(inner_channel, in_channels,
                      kernel_size=3, padding='same', bias=False),
            # nn.BatchNorm2d(in_channels),
            nn.PReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inner_channel, in_channels,
                      kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(in_channels, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU(inplace=True),
        )


    def forward(self, x):
        output = self.conv_layer(x) + x
        return output

class RDB_Block(nn.Module):
    def __init__(self, in_channels, inner_channel= [64,128,256]):
        super(RDB_Block, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, inner_channel[0],
                      kernel_size=3, padding='same', bias=False),
            # nn.BatchNorm2d(inner_channel[0]),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(negative_slope= 0.2,inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels + inner_channel[0], inner_channel[1],
                      kernel_size=3, padding='same', bias=False),
            # nn.BatchNorm2d(inner_channel[1]),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(negative_slope= 0.2,inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels + inner_channel[0] + inner_channel[1], inner_channel[2],
                      kernel_size=3, padding='same', bias=False),
            # nn.BatchNorm2d(inner_channel[2]),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(negative_slope= 0.2,inplace=True),
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels + inner_channel[0] + inner_channel[1]+ inner_channel[2], in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels, eps=1e-5, momentum=0.01, affine=True),
        )


    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x,x1), dim=1))
        x3 = self.conv3(torch.cat((x,x1,x2), dim=1))
        output = self.conv1x1(torch.cat((x, x1, x2, x3), dim=1)) + x
        # output = self.act_fn(output)
        return output


class low_frequecy_network(nn.Module):
    def __init__(self, in_channels,inner_channel = 64):
        super(low_frequecy_network, self).__init__()
        self.conv_in = nn.Conv2d(in_channels, inner_channel,
                      kernel_size=9, padding='same', bias=False)
        self.RB_1 = RB_Block(inner_channel)
        self.RB_2 = RB_Block(inner_channel)
        self.RB_3 = RB_Block(inner_channel)
        self.RB_4 = RB_Block(inner_channel)
        self.out_conv1 = nn.Conv2d(inner_channel, inner_channel,
                      kernel_size=9, padding='same', bias=False)
        self.out_conv2 = nn.Conv2d(inner_channel, in_channels,
                                   kernel_size=9, padding='same', bias=False)
        # self.act_fn = nn.Sigmoid()
        # self.act_fn = nn.LeakyReLU(inplace=True)
        self.act_fn = nn.GELU()

    def forward(self, x):
        x1 = self.conv_in(x)
        rb1 = self.RB_1(x1)
        rb2 = self.RB_2(rb1+x1)
        rb3 = self.RB_3(rb2+x1)
        rb4 = self.RB_4(rb3 + x1)
        x2 = self.out_conv1(rb4 + x1)
        out = self.out_conv2(x2 + x1)
        out = self.act_fn(out)

        return out,[rb1, rb2, rb3, rb4]


class high_frequecy_network(nn.Module):
    def __init__(self, in_channels,inner_channel = 64):
        super(high_frequecy_network, self).__init__()
        self.conv_in = nn.Conv2d(in_channels, inner_channel,
                      kernel_size=9, padding='same', bias=False)
        self.RDB_1 = RDB_Block(inner_channel)
        self.RDB_2 = RDB_Block(inner_channel)
        self.RDB_3 = RDB_Block(inner_channel)
        self.RDB_4 = RDB_Block(inner_channel)
        self.out_conv1x1 = nn.Conv2d(inner_channel * 4, inner_channel,
                                   kernel_size=1, bias=False)
        self.out_conv1 = nn.Conv2d(inner_channel, inner_channel,
                      kernel_size=9, padding='same', bias=False)
        self.out_conv2 = nn.Conv2d(inner_channel, in_channels,
                                   kernel_size=9, padding='same', bias=False)
        # self.act_fn = nn.LeakyReLU(inplace=True)
        self.act_fn = nn.Tanh()
        # self.act_fn = nn.GELU()

    def forward(self, x,rb):
        x1 = self.conv_in(x)
        rdb1 = self.RDB_1(x1 + rb[0])
        rdb2 = self.RDB_2(rdb1 + rb[1])
        rdb3 = self.RDB_3(rdb2 + rb[2])
        rdb4 = self.RDB_4(rdb3 + rb[3])
        rdb_out = torch.cat((rdb1, rdb2,rdb3,rdb4), dim=1)
        x2 = self.out_conv1x1(rdb_out)
        x3 = self.out_conv1(x2)
        out = self.out_conv2(x3 + x1)

        return self.act_fn(out)

class wavelet_net(nn.Module):
    def __init__(self,J=1, wave='db1', mode = 'zero',inner_channel=64):
        super(wavelet_net, self).__init__()
        self.low = low_frequecy_network(3,inner_channel)
        self.high = high_frequecy_network(9,inner_channel)
        # self.high = high_frequecy_network(3,64)
        self.sfm = SWTForward(J, wave, mode,requires_grad=True)
        self.ifm = SWTInverse(wave, mode,requires_grad=True)
    def forward(self,img):
        coeffs = self.sfm(img)[0]
        ll = coeffs[:,[0,4,8]]
        detail = coeffs[:,[1,2,3,5,6,7,9,10,11]]
        ll_out,rb_out = self.low(ll)
        detail_out = self.high(detail,rb_out)
        # ll_out,rb_out = self.low(img)
        # detail_out = self.high(img,rb_out)
        recon_R = self.ifm([torch.cat((ll_out[:,[0]],detail_out[:,0:3]),dim=1)])
        recon_G = self.ifm([torch.cat((ll_out[:,[1]],detail_out[:,3:6]),dim=1)])
        recon_B = self.ifm([torch.cat((ll_out[:,[2]],detail_out[:,6:9]),dim=1)])
        recon = torch.cat((recon_R, recon_G, recon_B), dim=1)
        return recon,ll_out,detail_out
        # return detail_out