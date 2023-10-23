import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import einops

class Attention(nn.Module):
    def __init__(self, ch, use_sn):
        super(Attention, self).__init__()
        # Channel multiplier
        self.ch = ch
        self.theta = nn.Conv2d(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.phi = nn.Conv2d(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.g = nn.Conv2d(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
        self.o = nn.Conv2d(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
        if use_sn:
            self.theta = spectral_norm(self.theta)
            self.phi = spectral_norm(self.phi)
            self.g = spectral_norm(self.g)
            self.o = spectral_norm(self.o)
        # Learnable gain parameter
        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, x, y=None):
        # Apply convs
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2,2])
        g = F.max_pool2d(self.g(x), [2,2])
        # Perform reshapes
        theta = theta.view(-1, self. ch // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self. ch // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self. ch // 2, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
        return self.gamma * o + x


class SPADE(nn.Module):
    def __init__(self, input_nc=48, hidden_nc=48, output_nc=48):
        super().__init__()

        self.input_nc = input_nc
        self.hidden_nc = hidden_nc
        self.output_nc = output_nc
        # self.ks = 3
        # self.pw = self.ks // 2
        # self.pad_type = 'nozero'
        # self.param_free_norm = nn.InstanceNorm2d(hidden_nc, affine=False)
        self.param_free_norm = nn.LayerNorm(hidden_nc)

        # The dimension of the intermediate embedding space. Yes, hardcoded.

        # self.mlp_shared = nn.Sequential(
        #         nn.Conv2d(input_nc, hidden_nc, kernel_size=self.ks, padding=self.pw),
        #         nn.ReLU()
        #     )
        # self.mlp_gamma = nn.Conv2d(hidden_nc, hidden_nc, kernel_size=self.ks, padding=self.pw)
        # self.mlp_beta = nn.Conv2d(hidden_nc, hidden_nc, kernel_size=self.ks, padding=self.pw)

        self.mlp_shared = nn.Sequential(
            nn.Linear(input_nc, hidden_nc),
            nn.LayerNorm(hidden_nc),
            nn.LeakyReLU()
            )
        # self.mlp_gamma = nn.Linear(hidden_nc, hidden_nc)
        # self.mlp_beta = nn.Linear(hidden_nc, hidden_nc)

        # self.mlp_content = nn.Sequential(nn.Linear(hidden_nc, hidden_nc), nn.LayerNorm(hidden_nc))
        # self.mlp_gamma = nn.Sequential(nn.Linear(hidden_nc, hidden_nc), nn.LayerNorm(hidden_nc))
        # self.mlp_beta = nn.Sequential(nn.Linear(hidden_nc, hidden_nc), nn.LayerNorm(hidden_nc))
        # self.mlp_out = nn.Sequential(nn.Linear(hidden_nc, hidden_nc), nn.LayerNorm(hidden_nc))

        self.mlp_gamma = nn.Sequential(nn.Linear(hidden_nc, hidden_nc))
        self.mlp_beta = nn.Sequential(nn.Linear(hidden_nc, hidden_nc))


    def forward(self, x, style_vector):


        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        actv = self.mlp_shared(style_vector)

        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # icon 128 ex1
        # relation = torch.sum(normalized * gamma, -1, keepdim=True)
        # relation = torch.sigmoid(relation)
        # relation = relation.expand_as(style_vector)
        # aggregated_features = relation * beta
        # out = normalized + aggregated_features + beta

        out = normalized * (1 + gamma) + beta

        return out



class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, use_se=False):
        super().__init__()
        # Attributes
        # self.learned_shortcut = (fin != fout)
        self.input_nc = fin
        self.hidden_nc = fout
        self.output_nc = fout
        # self.use_se = use_se
        # self.conv_0 = nn.Conv2d(self.input_nc, self.hidden_nc, kernel_size=3, padding=dilation, dilation=dilation)
        # self.conv_1 = nn.Conv2d(self.hidden_nc, self.hidden_nc, kernel_size=3, padding=dilation, dilation=dilation)
        # if self.learned_shortcut:
        #     self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        self.conv_0 = nn.Linear(self.hidden_nc, self.hidden_nc)
        self.conv_1 = nn.Linear(self.hidden_nc, self.hidden_nc)

        self.norm_0 = SPADE(input_nc=self.input_nc, hidden_nc=self.hidden_nc, output_nc=self.hidden_nc)
        self.norm_1 = SPADE(input_nc=self.hidden_nc, hidden_nc=self.hidden_nc, output_nc=self.hidden_nc)

        # if self.learned_shortcut:
        #     self.norm_s = SPADE(spade_config_str, fin, ic, PONO=opt.PONO, use_apex=opt.apex)
        # if use_se:
        #     self.se_layar = SELayer(fout)

    # the style_vector as input
    def forward(self, x, style_vector):
        # x_s = self.shortcut(x, seg1)

        dx = self.conv_0(self.actvn(self.norm_0(x, style_vector)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, style_vector)))
        # if self.use_se:
        #     dx = self.se_layar(dx)
        out = x + dx
        return out

    def shortcut(self, x, seg1):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg1))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        input_channel = 64
        self.nf = 48
        output_channel = 3

        def _mapping_layers():
            yield nn.Linear(input_channel, self.nf)
            yield nn.LayerNorm(self.nf)
            yield nn.LeakyReLU()

        self.mapping = nn.Sequential(*_mapping_layers())

        self.spade_1 = SPADEResnetBlock(self.nf, self.nf)
        self.spade_2 = SPADEResnetBlock(self.nf, self.nf)
        self.attn = Attention(self.nf, False)
        self.spade_3 = SPADEResnetBlock(self.nf, self.nf)
        self.spade_4 = SPADEResnetBlock(self.nf, self.nf)

        self.linear = nn.Linear(self.nf, output_channel)
        self.actn = nn.Tanh()


    def forward(self, input, style_vector, B, H, W):

        D = self.nf

        x = self.mapping(input)  # (BHW, D)

        x = self.spade_1(x, style_vector)
        x = self.spade_2(x, style_vector)

        x = einops.rearrange(x, '(B H W) D -> B D H W', B=B, D=D, H=H, W=W)
        x = self.attn(x)
        x = einops.rearrange(x, 'B D H W -> (B H W) D')
        x = self.spade_3(x, style_vector)
        x = self.spade_4(x, style_vector)

        x = self.linear(x)
        x = self.actn(x)

        return x
