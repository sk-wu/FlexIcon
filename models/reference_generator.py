import torch.nn as nn
import torch
import torch.nn.utils.spectral_norm as spectral_norm
import torch.nn.functional as F
from typing import Callable
from .rescae import get_residual_unet
from .resnet import get_resnet_by_depth
from .decoder import Decoder
import einops


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
    def __init__(self, spec_norm=False, LR=0.2):
        super(ReferenceGenerator, self).__init__()

        contour_channels = 1
        image_channels = 3
        embedding_dim = 16
        style_dim = 48
        content_encoder_arch = 'M51'
        content_extractor_arch = 'S31'
        norm_conv_channels = 16
        resnet_depth = 50
        decoder_width = 32
        decoder_depth = 4

        self.contour_channels = contour_channels
        self.image_channels = image_channels
        self.embedding_dim = embedding_dim
        self.style_dim = style_dim
        self.content_encoder_arch = content_encoder_arch
        self.content_extractor_arch = content_extractor_arch
        self.norm_conv_channels = norm_conv_channels
        self.resnet_depth = resnet_depth
        self.decoder_width = decoder_width
        self.decoder_depth = decoder_depth

        self.content_encoder = nn.Sequential(
            get_residual_unet(contour_channels, embedding_dim, content_encoder_arch),
            # MMOEContentEncoder(),
            nn.Tanh())

        self.content_extractor = nn.Sequential(
            NormConv2d(norm_conv_channels),
            get_residual_unet(norm_conv_channels, contour_channels, content_extractor_arch),
            nn.Tanh(),
        )

        self.style_encoder = nn.Sequential(
            get_resnet_by_depth(resnet_depth, num_classes=style_dim, in_channels=image_channels),
            nn.Tanh()
        )

        self.decoder = Decoder()

    def extract_content(self, x: torch.Tensor) -> torch.Tensor:
        return self.content_extractor(x)

    def encode_content(self, c: torch.Tensor) -> torch.Tensor:
        return self.content_encoder(c)

    def encode_style(self, x: torch.Tensor) -> torch.Tensor:
        return self.style_encoder(x)

    def decode(self, e: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        B, _, H, W = e.shape

        s = einops.repeat(s, 'B S -> B S H W', B=B, H=H, W=W)
        h = torch.cat([e, s], 1)

        h = einops.rearrange(h, 'B D H W -> (B H W) D')

        style_vector = einops.rearrange(s, 'B S H W -> (B H W) S')

        r = self.decoder(h, style_vector, B, H, W)
        r = einops.rearrange(r, '(B H W) C -> B C H W', B=B, H=H, W=W)

        return r

    def forward(self, c, x, color_theme=None, tps_reference=None):  # c is line art, x is reference image
        # Reconstruction Error
        ext1 = self.encode_content(c)  # content
        style1 = self.encode_style(x)  # style
        style2 = style1.roll(1, 0).contiguous()  # style
        rec11 = self.decode(ext1, style1)  # aligned
        rec12 = self.decode(ext1, style2)  # unaligned

        # Extraction Error
        contour_x = self.extract_content(x)

        # Content Consistency
        # self.content_extractor.requires_grad_(False)
        # contour12 = self.extract_content(rec12)
        # self.content_extractor.requires_grad_(True)

        unaligned_style_image = x.detach().roll(1, 0).contiguous()

        # return rec11, rec12, contour_x, contour12, unaligned_style_image
        return rec11, rec12, contour_x, unaligned_style_image

    def forward_inference(self, c, x, color_theme=None, tps_reference=None):  # c is line art, x is reference image
        # Reconstruction Error
        ext1 = self.encode_content(c)  # content
        style1 = self.encode_style(x)  # style
        rec11 = self.decode(ext1, style1)  # aligned

        return rec11


