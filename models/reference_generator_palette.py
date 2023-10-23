import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
import torch.nn.functional as F
from typing import Callable
import einops


class ThemeMapping(nn.Module):
    def __init__(self, in_dim=3, out_dim=64):
        super(ThemeMapping, self).__init__()
        self.increase_dimension = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(out_dim, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc1 = nn.Linear(192, out_dim*2)
        self.layernorm1 = nn.LayerNorm(out_dim*2)
        self.fc2 = nn.Linear(out_dim*2, out_dim)
        self.layernorm2 = nn.LayerNorm(out_dim)
        self.act = nn.LeakyReLU()

    def forward(self, color_theme):  # color theme (B, 3, 1, 3)
        x = self.increase_dimension(color_theme)  # color theme (B, 64, 1, 3)
        B, C, H, W = x.size()  # color theme (B, 64, 1, 3)
        x = x.reshape(B, -1, C*H*W)  # (B, 1, 192)
        x = self.fc1(x)  # (B, 1, 128)
        x = self.layernorm1(x)  # (B, 1, 128)
        x = self.act(x)  # (B, 1, 128)
        x = self.fc2(x)  # (B, 1, 64)
        x = self.layernorm2(x)
        return x


class Lambda(nn.Module):
    def __init__(self, func: Callable):
        self.func = func
        super().__init__()

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class NormConv2d(nn.Conv2d):
    def __init__(self, channels, kernel_size=2):
        super().__init__(1, channels, kernel_size,
                         padding='same',
                         padding_mode='replicate',
                         bias=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        B = input.shape[0]
        input = einops.rearrange(input, 'B C H W -> (B C) 1 H W')

        weight = self.weight - einops.reduce(self.weight, 'K 1 H W -> K 1 1 1', 'mean')
        output = self._conv_forward(input, weight, self.bias)
        output = einops.rearrange(output, '(B C) K H W -> B C K H W', B=B)

        output = torch.abs(output)
        output = einops.reduce(output, 'B C K H W -> B K H W', 'mean')

        return output


class ConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out, spec_norm=False, LR=0.01, stride=1, up=False):
        super(ConvBlock, self).__init__()

        self.up = up
        if self.up:
            self.up_smaple = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            self.up_smaple = None

        if spec_norm:
            self.main = nn.Sequential(
                spectral_norm(nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=stride, padding=1, bias=False)),
                nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
                nn.LeakyReLU(LR, inplace=True),
            )

        else:
            self.main = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
                nn.LeakyReLU(LR, inplace=True),
            )

    def forward(self, x1, x2=None):
        if self.up_smaple is not None:
            x1 = self.up_smaple(x1)
            # input is CHW
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            # if you have padding issues, see
            # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
            # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
            x = torch.cat([x2, x1], dim=1)
            return self.main(x)
        else:
            return self.main(x1)


class Encoder(nn.Module):
    """Discriminator network with PatchGAN.
    W = (W - F + 2P) /S + 1"""

    def __init__(self, in_channels=1, spec_norm=False, LR=0.2):
        super(Encoder, self).__init__()

        self.layer1 = ConvBlock(in_channels, 16, spec_norm, LR=LR) # 256
        self.layer2 = ConvBlock(16, 16, spec_norm, LR=LR) # 256
        self.layer3 = ConvBlock(16, 16, spec_norm, LR=LR) # 128
        self.layer4 = ConvBlock(16, 16, spec_norm, LR=LR) # 128
        self.layer5 = ConvBlock(16, 16, spec_norm, LR=LR) # 64
        self.layer6 = ConvBlock(16, 16, spec_norm, LR=LR) # 64
        self.layer7 = ConvBlock(16, 16, spec_norm, LR=LR) # 32
        self.layer8 = ConvBlock(16, 16, spec_norm, LR=LR) # 32
        self.layer9 = ConvBlock(16, 16, spec_norm, LR=LR) # 16
        self.layer10 = ConvBlock(16, 16, spec_norm, LR=LR) # 16
        self.last_conv = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        feature_map1 = self.layer1(x)
        feature_map2 = self.layer2(feature_map1)
        feature_map3 = self.layer3(feature_map2)
        feature_map4 = self.layer4(feature_map3)
        feature_map5 = self.layer5(feature_map4)
        feature_map6 = self.layer6(feature_map5)
        feature_map7 = self.layer7(feature_map6)
        feature_map8 = self.layer8(feature_map7)
        feature_map9 = self.layer9(feature_map8)
        feature_map10 = self.layer10(feature_map9)
        output = feature_map10
        output = self.last_conv(output)

        return output


class ReferenceGenerator(nn.Module):
    def __init__(self, in_dim=3, color_num=3, mid_dim=144, style_dim=48):
        super(ReferenceGenerator, self).__init__()

        self.model = nn.Sequential(
                     nn.Linear(in_dim*color_num, mid_dim),
                     nn.LayerNorm(mid_dim),
                     nn.LeakyReLU(),
                     nn.Linear(mid_dim, mid_dim),
                     nn.LayerNorm(mid_dim),
                     nn.LeakyReLU(),
                     nn.Linear(mid_dim, style_dim*2),
                     nn.LayerNorm(style_dim*2),
                     nn.LeakyReLU(),
                     nn.Linear(style_dim*2, style_dim),
                     nn.Tanh(),
                    )

    def forward(self, color_theme):  # color theme (B, 3, 1, 5)
        B, C, H, W = color_theme.size()  # color theme (B, 3, 1, 5)
        x = color_theme.reshape(B, -1)  # (B, -1)
        x = self.model(x)  # (B, 48)
        return x

    # def forward_interpolation(self, color_theme):  # color theme (B, 3, 1, 5)
    #     B, C, H, W = color_theme[0].size()  # color theme (B, 3, 1, 5)
    #     x1 = color_theme[0].reshape(B, -1)  # (B, -1)
    #     x1 = self.model(x1)  # (B, 48)
    #
    #     x2 = color_theme[1].reshape(B, -1)
    #     x2 = self.model(x2)
    #
    #     x = x1 * 0.2 + x2 * 0.8
    #     x = torch.clip(x, -1, 1)
    #
    #     return x


