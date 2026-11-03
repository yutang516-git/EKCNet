
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
from einops import rearrange


def init_weights(net, init_type='xavier_uniform_', gain=1.0):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier_normal_':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'xavier_uniform_':
                init.xavier_uniform_(m.weight.data, gain=gain)
            elif init_type == 'kaiming_normal_':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'kaiming_uniform_':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1 or classname.find('GroupNorm') != -1:
            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


def param_network(model):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print("The number of parameters: {}".format(num_params))


class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=1):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, scale_factor=2):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor),
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        if downsample is not None:
            self.downsample = nn.Sequential(
                nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes),
            )
        self.stride = stride
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Encoder(nn.Module):
    def __init__(self, inChannel=2, baseChannel=24):
        super(Encoder, self).__init__()
        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1 = nn.Sequential(
            BasicBlock(inChannel, baseChannel, stride=1, downsample=True),
            BasicBlock(baseChannel, baseChannel, stride=1, downsample=None)
        )
        self.Conv2 = nn.Sequential(
            BasicBlock(baseChannel, baseChannel * 2, stride=1, downsample=True),
            BasicBlock(baseChannel * 2, baseChannel * 2, stride=1, downsample=None)
        )
        self.Conv3 = nn.Sequential(
            BasicBlock(baseChannel * 2, baseChannel * 4, stride=1, downsample=True),
            BasicBlock(baseChannel * 4, baseChannel * 4, stride=1, downsample=None)
        )
        self.Conv4 = nn.Sequential(
            BasicBlock(baseChannel * 4, baseChannel * 8, stride=1, downsample=True),
            BasicBlock(baseChannel * 8, baseChannel * 8, stride=1, downsample=None)
        )
        self.Conv5 = nn.Sequential(
            BasicBlock(baseChannel * 8, baseChannel * 16, stride=1, downsample=True),
            BasicBlock(baseChannel * 16, baseChannel * 16, stride=1, downsample=None)
        )


    def forward_first_layer(self, x):
        """只执行第一层，返回 x1，用于在浅层进行跨分支信息交换。"""
        return self.Conv1(x)

    def forward_rest(self, x1, first_injection=None, adapters=None):

        if first_injection is not None:
            x1 = x1 + first_injection

        # idx=1 (64³)：adapter 输入是 x1
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        if adapters is not None and adapters[1] is not None:
            x2 = x2 + adapters[1](x1)

        # idx=2 (32³)：adapter 输入是 x2
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        if adapters is not None and adapters[2] is not None:
            x3 = x3 + adapters[2](x2)

        # idx=3 (16³)：adapter 输入是 x3
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        if adapters is not None and adapters[3] is not None:
            x4 = x4 + adapters[3](x3)

        # idx=4 (8³)：adapter 输入是 x4
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        if adapters is not None and adapters[4] is not None:
            x5 = x5 + adapters[4](x4)

        return x1, x2, x3, x4, x5

    def forward(self, x, first_injection=None):

        x1 = self.forward_first_layer(x)
        return self.forward_rest(x1, first_injection=first_injection)

# ====================== Dicro chasp ======================
class IntraModalCoAttention(nn.Module):

    def __init__(self, channels, reduction=16, groups=2):
        super().__init__()
        self.groups = groups

        # 通道注意力
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 空间注意力
        self.spatial_attn = nn.Sequential(
            nn.Conv3d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

    def _channel_shuffle(self, x, groups):

        B, C, D, H, W = x.shape
        assert C % groups == 0, f"Channels {C} must be divisible by groups {groups}"
        x = x.view(B, groups, C // groups, D, H, W)
        x = x.transpose(1, 2).contiguous()
        x = x.view(B, C, D, H, W)
        return x

    def forward(self, x_A, x_B):

        x_cat = torch.cat([x_A, x_B], dim=1)
        x_shuffled = self._channel_shuffle(x_cat, self.groups)
        x_chan, x_spat = x_shuffled.chunk(2, dim=1)


        attn_chan = self.fc(self.global_pool(x_chan))
        attn_spat = self.spatial_attn(x_spat)


        h_A = x_A + self.alpha * (attn_chan * x_B)
        h_B = x_B + self.beta * (attn_spat * x_A)

        return h_A, h_B


class DifferentialCrossAttention(nn.Module):

    def __init__(self, hidden_size, input_size=(8,8,8), num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v1_proj = nn.Linear(hidden_size, hidden_size)
        self.v2_proj = nn.Linear(hidden_size, hidden_size)

        self.out1 = nn.Linear(hidden_size, hidden_size)
        self.out2 = nn.Linear(hidden_size, hidden_size)

        self.gamma = nn.Parameter(torch.tensor(0.5))
        self.delta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x1, x2):
        B, C, D, H, W = x1.shape
        N = D * H * W

        x1_flat = x1.flatten(2).transpose(1, 2)
        x2_flat = x2.flatten(2).transpose(1, 2)

        Q = self.q_proj(x1_flat)
        K = self.k_proj(x2_flat)
        V1 = self.v1_proj(x1_flat)
        V2 = self.v2_proj(x2_flat)

        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V1 = V1.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V2 = V2.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_probs = attn.softmax(dim=-1)
        h12 = torch.matmul(attn_probs, V2)
        h12 = h12.transpose(1, 2).contiguous().view(B, N, C)
        h12 = self.out1(h12)


        Q2 = self.q_proj(x2_flat)
        K1 = self.k_proj(x1_flat).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V1_ = V1
        Q2 = Q2.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        attn2 = torch.matmul(Q2, K1.transpose(-2, -1)) * self.scale
        attn_probs2 = attn2.softmax(dim=-1)
        h21 = torch.matmul(attn_probs2, V1_)
        h21 = h21.transpose(1, 2).contiguous().view(B, N, C)
        h21 = self.out2(h21)

        h12 = h12.transpose(1, 2).view(B, C, D, H, W)
        h21 = h21.transpose(1, 2).view(B, C, D, H, W)

        diff = h12 - h21
        h12_enh = h12 + self.gamma * diff
        h21_enh = h21 + self.delta * diff
        return h12_enh, h21_enh


class ImprovedBottleneckInteraction(nn.Module):

    def __init__(self, hidden_size, input_size=(8,8,8), num_heads=8):
        super().__init__()
        self.isca = IntraModalCoAttention(hidden_size)
        self.diff_cross = DifferentialCrossAttention(hidden_size, input_size, num_heads)

    def forward(self, x_A, x_B):
        h11, h22 = self.isca(x_A, x_B)
        h12_enh, h21_enh = self.diff_cross(x_A, x_B)
        x5_new = torch.cat([h11, h12_enh, h21_enh, h22], dim=1)
        return x5_new


class ShallowInjection(nn.Module):

    def __init__(self, channels, gate_bias=-2.0, init_scale=1.0):
        super(ShallowInjection, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(channels),
        )
        self.gate = nn.Conv3d(channels, 1, kernel_size=1)
        nn.init.constant_(self.gate.bias, gate_bias)

        # 可学习的整体缩放因子
        self.scale = nn.Parameter(torch.tensor(init_scale))

    def forward(self, src_feat):
        g = torch.sigmoid(self.gate(src_feat))   # [B,1,D,H,W]
        return self.scale * g * self.proj(src_feat)
class EKCNet(nn.Module):
    def __init__(self, inChannel=2, outChannel=4, baseChannel=16,gate_bias=-2.0):
        super(EKCNet, self).__init__()
        self.encoder1 = Encoder(inChannel=inChannel, baseChannel=baseChannel)
        self.encoder2 = Encoder(inChannel=inChannel, baseChannel=baseChannel)

        self.bottleneck = ImprovedBottleneckInteraction(
            hidden_size=baseChannel * 16,
            input_size=(8,8,8),
            num_heads=8
        )

        self.injection_1to2 = ShallowInjection(baseChannel, gate_bias=gate_bias)
        self.injection_2to1 = ShallowInjection(baseChannel, gate_bias=gate_bias)

        self.Up5 = up_conv(ch_in=baseChannel * 16 * 4, ch_out=baseChannel * 8)
        self.Up_conv5 = BasicBlock(baseChannel * 24, baseChannel * 8, stride=1, downsample=True)  # 修正为24

        self.Up4 = up_conv(ch_in=baseChannel * 8, ch_out=baseChannel * 4)
        self.Up_conv4 = BasicBlock(baseChannel * 12, baseChannel * 4, stride=1, downsample=True)

        self.Up3 = up_conv(ch_in=baseChannel * 4, ch_out=baseChannel * 2)
        self.Up_conv3 = BasicBlock(baseChannel * 6, baseChannel * 2, stride=1, downsample=True)

        self.Up2 = up_conv(ch_in=baseChannel * 2, ch_out=baseChannel)
        self.Up_conv2 = BasicBlock(baseChannel * 3, baseChannel, stride=1, downsample=True)

        self.Conv_1x1 = nn.Conv3d(baseChannel, outChannel, kernel_size=1, stride=1, padding=0)

    def forward(self, x, return_enc2_feats=False):
        flair, t1, t1ce, t2 = x[:, :1], x[:, 1:2], x[:, 2:3], x[:, 3:]

        x_in1 = torch.cat((t1, t1ce), dim=1)
        x_in2 = torch.cat((flair, t2), dim=1)

        x1_1 = self.encoder1.forward_first_layer(x_in1)
        x1_2 = self.encoder2.forward_first_layer(x_in2)

        inj_1to2 = self.injection_1to2(x1_1)  # encoder1 -> encoder2
        inj_2to1 = self.injection_2to1(x1_2)  # encoder2 -> encoder1

        h_out_1 = self.encoder1.forward_rest(x1_1, first_injection=inj_2to1)
        h_out_2 = self.encoder2.forward_rest(x1_2, first_injection=inj_1to2)

        x1 = torch.cat((h_out_1[0], h_out_2[0]), dim=1)
        x2 = torch.cat((h_out_1[1], h_out_2[1]), dim=1)
        x3 = torch.cat((h_out_1[2], h_out_2[2]), dim=1)
        x4 = torch.cat((h_out_1[3], h_out_2[3]), dim=1)
        x5 = self.bottleneck(h_out_1[4], h_out_2[4])

        d5 = self.Up5(x5);
        d5 = torch.cat((x4, d5), dim=1);
        d5 = self.Up_conv5(d5)
        d4 = self.Up4(d5);
        d4 = torch.cat((x3, d4), dim=1);
        d4 = self.Up_conv4(d4)
        d3 = self.Up3(d4);
        d3 = torch.cat((x2, d3), dim=1);
        d3 = self.Up_conv3(d3)
        d2 = self.Up2(d3);
        d2 = torch.cat((x1, d2), dim=1);
        d2 = self.Up_conv2(d2)

        out = self.Conv_1x1(d2)

        if return_enc2_feats:
            return out, h_out_2
        return out



if __name__ == "__main__":
    model = EKCNet(inChannel=2, outChannel=4)
    x = torch.ones((2, 4, 128, 128, 128))
    output = model(x)
    print("Output shape:", output.shape)
    param_network(model)