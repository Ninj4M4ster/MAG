import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x_float = x.float() 
        mean = x_float.mean(dim=[1, 2, 3], keepdim=True)
        var = x_float.var(dim=[1, 2, 3], unbiased=False, keepdim=True)
        x_norm = (x_float - mean) / torch.sqrt(var + self.eps)
        x_norm = x_norm.type_as(x) 

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x_out = x_norm * self.gamma.view(*shape) + self.beta.view(*shape)
            return x_out
        else:
            return x_norm

class SafeInstanceNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=False):
        super(SafeInstanceNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

    def forward(self, x):
        x_float = x.float()
        out = F.instance_norm(x_float, running_mean=None, running_var=None, 
                              weight=None, bias=None, use_input_stats=True, momentum=0.1, eps=self.eps)
        return out.type_as(x)


pad_dict = dict(
     zero = nn.ZeroPad2d,
  reflect = nn.ReflectionPad2d,
replicate = nn.ReplicationPad2d)

conv_dict = dict(
   conv2d = nn.Conv2d,
 deconv2d = nn.ConvTranspose2d)

norm_dict = dict(
     none = lambda x: lambda x: x,
 spectral = lambda x: lambda x: x,
    batch = nn.BatchNorm2d,
 instance = SafeInstanceNorm,
    layer = LayerNorm)

activ_dict = dict(
      none = lambda: lambda x: x,
    #   relu = lambda: nn.ReLU(inplace=True),
      relu = lambda: nn.ReLU(inplace=False),
    #  lrelu = lambda: nn.LeakyReLU(0.2, inplace=True),
     lrelu = lambda: nn.LeakyReLU(0.2, inplace=False),
     prelu = lambda: nn.PReLU(),
    #   selu = lambda: nn.SELU(inplace=True),
      selu = lambda: nn.SELU(inplace=False),
      tanh = lambda: nn.Tanh())


class ConvolutionBlock(nn.Module):
    def __init__(self, conv='conv2d', norm='instance', activ='relu', pad='reflect', padding=0, **conv_opts):
        super(ConvolutionBlock, self).__init__()

        self.pad = pad_dict[pad](padding)
        self.conv = conv_dict[conv](**conv_opts)

        out_channels = conv_opts['out_channels']
        self.norm = norm_dict[norm](out_channels)
        if norm == "spectral": self.conv = spectral_norm(self.conv)

        self.activ = activ_dict[activ]()

    def forward(self, x):
        return self.activ(self.norm(self.conv(self.pad(x))))


class ResidualBlock(nn.Module):
    def __init__(self, channels, norm='instance', activ='relu', pad='reflect'):
        super(ResidualBlock, self).__init__()

        block = []
        block += [ConvolutionBlock(
            in_channels=channels, out_channels=channels, kernel_size=3,
            stride=1, padding=1, norm=norm, activ=activ, pad=pad)]
        block += [ConvolutionBlock(
            in_channels=channels, out_channels=channels, kernel_size=3,
            stride=1, padding=1, norm=norm, activ='none', pad=pad)]
        self.model = nn.Sequential(*block)

    def forward(self, x): return self.model(x) + x


class FullyConnectedBlock(nn.Module):
    def __init__(self, input_ch, output_ch, norm='none', activ='relu'):
        super(FullyConnectedBlock, self).__init__()

        self.fc = nn.Linear(input_ch, output_ch, bias=True)
        self.norm = norm_dict[norm](output_ch)
        if norm == "spectral": self.fc = spectral_norm(self.fc)
        self.activ = activ_dict[activ]()

    def forward(self, x): return self.activ(self.norm(self.fc(x)))
