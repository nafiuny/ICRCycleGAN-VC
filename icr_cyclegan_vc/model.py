"""
Trains ICRCycleGAN-VC as described in 
Inspired by https://github.com/GANtastic3/MaskCycleGAN-VC
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class GLU(nn.Module):
    """Custom implementation of GLU since the paper assumes GLU won't reduce
    the dimension of tensor by 2.
    """

    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class PixelShuffle(nn.Module):
    """Custom implementation pf Pixel Shuffle since PyTorch's PixelShuffle
    requires a 4D input (we have 3D inputs).
    """

    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        n = x.shape[0]
        c_out = x.shape[1] // 2
        w_new = x.shape[2] * 2
        return x.view(n, c_out, w_new)


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        
        self.conv = nn.Conv1d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.InstanceNorm1d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)




class Inception(nn.Module):
    """Inception-ResNet Blocks  of the Generator
    """
    def __init__(self):
        super(Inception, self).__init__()
        
        self.branch1x1 = BasicConv2d(256, 256, kernel_size=1, padding=0)
        self.branch3x3_1 = BasicConv2d(256, 256, kernel_size=1, padding=1)
        self.branch3x3_2 = BasicConv2d(256, 256, kernel_size=3, padding=0)
        self.branch5x5_1 = BasicConv2d(256, 256, kernel_size=1, padding=1)
        self.branch5x5_2 = BasicConv2d(256, 256, kernel_size=5, padding=1)
        self.branch_pool = BasicConv2d(256, 256, kernel_size=1, padding=0)
        
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = branch1x1 + branch3x3 + branch5x5 + branch_pool
        return outputs + x
        
class DownSampleGenerator(nn.Module):
    """Downsampling blocks of the Generator.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DownSampleGenerator, self).__init__()

        self.convLayer = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding),
                                       nn.InstanceNorm2d(num_features=out_channels,
                                                         affine=True))
        self.convLayer_gates = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                       out_channels=out_channels,
                                                       kernel_size=kernel_size,
                                                       stride=stride,
                                                       padding=padding),
                                             nn.InstanceNorm2d(num_features=out_channels,
                                                               affine=True))

    def forward(self, x):
        # GLU
        return self.convLayer(x) * torch.sigmoid(self.convLayer_gates(x))


class Generator(nn.Module):
    """Generator of ICRCycleGAN-VC
    """

    def __init__(self, input_shape=(80, 64), residual_in_channels=256):
        super(Generator, self).__init__()
        Cx, Tx = input_shape
        self.flattened_channels = (Cx // 4) * residual_in_channels

        # 2D Conv Layer
        self.conv1 = nn.Conv2d(in_channels=2,
                               out_channels=residual_in_channels // 2,
                               kernel_size=(5, 15),
                               stride=(1, 1),
                               padding=(2, 7))

        self.conv1_gates = nn.Conv2d(in_channels=2,
                                     out_channels=residual_in_channels // 2,
                                     kernel_size=(5, 15),
                                     stride=1,
                                     padding=(2, 7))

        # 2D Downsampling Layers
        self.downSample1 = DownSampleGenerator(in_channels=residual_in_channels // 2,
                                               out_channels=residual_in_channels,
                                               kernel_size=5,
                                               stride=2,
                                               padding=2)

        self.downSample2 = DownSampleGenerator(in_channels=residual_in_channels,
                                               out_channels=residual_in_channels,
                                               kernel_size=5,
                                               stride=2,
                                               padding=2)

        # 2D -> 1D Conv
        self.conv2dto1dLayer = nn.Conv1d(in_channels=self.flattened_channels,
                                         out_channels=residual_in_channels,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)

        self.conv2dto1dLayer_tfan = nn.InstanceNorm1d(
            num_features=residual_in_channels, affine=True)

        # Inception-ResNet Blocks
        self.Inception1 = Inception()
        self.Inception2 = Inception()
        self.Inception3 = Inception()
        self.Inception4 = Inception()
        self.Inception5 = Inception()

        # 1D -> 2D Conv
        self.conv1dto2dLayer = nn.Conv1d(in_channels=residual_in_channels,
                                         out_channels=self.flattened_channels,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)
        self.conv1dto2dLayer_tfan = nn.InstanceNorm1d(
            num_features=self.flattened_channels, affine=True)

        # UpSampling Layers
        self.upSample1 = self.upsample(in_channels=residual_in_channels,
                                       out_channels=residual_in_channels * 4,
                                       kernel_size=5,
                                       stride=1,
                                       padding=2)

        
        self.upSample2 = self.upsample(in_channels=residual_in_channels,
                                       out_channels=residual_in_channels * 2,
                                       kernel_size=5,
                                       stride=1,
                                       padding=2)

        # 2D Conv Layer
        self.lastConvLayer = nn.Conv2d(in_channels=residual_in_channels // 2,
                                       out_channels=1,
                                       kernel_size=(5, 15),
                                       stride=(1, 1),
                                       padding=(2, 7))

    def downsample(self, in_channels, out_channels, kernel_size, stride, padding):
        self.ConvLayer = nn.Sequential(nn.Conv1d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding),
                                       nn.InstanceNorm1d(
                                           num_features=out_channels,
                                           affine=True),
                                       nn.ReLU())

        return self.ConvLayer

    def upsample(self, in_channels, out_channels, kernel_size, stride, padding):
        self.convLayer = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding),
                                       nn.PixelShuffle(upscale_factor=2),
                                       nn.InstanceNorm2d(
                                           num_features=out_channels // 4,
                                           affine=True),
                                       nn.ReLU())
        return self.convLayer

    def forward(self, x, mask):
        # Conv2d
        x = torch.stack((x*mask, mask), dim=1)
        conv1 = self.conv1(x) * torch.sigmoid(self.conv1_gates(x))  # GLU
        
	# Downsampling
        downsample1 = self.downSample1(conv1)
        downsample2 = self.downSample2(downsample1)
        
        # Reshape
        reshape2dto1d = downsample2.view(
            downsample2.size(0), self.flattened_channels, 1, -1)
        reshape2dto1d = reshape2dto1d.squeeze(2)
        
        # 2D -> 1D
        conv2dto1d_layer = self.conv2dto1dLayer(reshape2dto1d)
        conv2dto1d_layer = self.conv2dto1dLayer_tfan(conv2dto1d_layer)
                
        # Inception-ResNet
        x1 = self.Inception1(conv2dto1d_layer)
        x1 = self.Inception2(x1)
        x1 = self.Inception3(x1)
        x1 = self.Inception4(x1)
        x1 = self.Inception5(x1)
        
        # 1D -> 2D
        conv1dto2d_layer = self.conv1dto2dLayer(x1)
        conv1dto2d_layer = self.conv1dto2dLayer_tfan(conv1dto2d_layer)
        
        # Reshape
        reshape1dto2d = conv1dto2d_layer.unsqueeze(2)
        reshape1dto2d = reshape1dto2d.view(reshape1dto2d.size(0), 256, 20, -1)
        

        # UpSampling
        upsample_layer_1 = self.upSample1(reshape1dto2d)
        upsample_layer_2 = self.upSample2(upsample_layer_1)

        # Conv2d
        output = self.lastConvLayer(upsample_layer_2)
        output = output.squeeze(1)
        return output


class Discriminator(nn.Module):
    """PatchGAN discriminator.
    """

    def __init__(self, input_shape=(80, 64), residual_in_channels=256):
        super(Discriminator, self).__init__()

        self.convLayer1 = nn.Sequential(nn.Conv2d(in_channels=1,
                                                  out_channels=residual_in_channels // 2,
                                                  kernel_size=(3, 3),
                                                  stride=(1, 1),
                                                  padding=(1, 1)),
                                        nn.ReLU())

        # Downsampling Layers
        self.downSample1 = self.downsample(in_channels=residual_in_channels // 2,
                                           out_channels=residual_in_channels,
                                           kernel_size=(3, 3),
                                           stride=(2, 2),
                                           padding=1)

        self.downSample2 = self.downsample(in_channels=residual_in_channels,
                                           out_channels=residual_in_channels * 2,
                                           kernel_size=(3, 3),
                                           stride=[2, 2],
                                           padding=1)

        self.downSample3 = self.downsample(in_channels=residual_in_channels * 2,
                                           out_channels=residual_in_channels * 4,
                                           kernel_size=[3, 3],
                                           stride=[2, 2],
                                           padding=1)


        # Conv Layer
        self.outputConvLayer = nn.Sequential(nn.Conv2d(in_channels=residual_in_channels * 4,
                                                       out_channels=1,
                                                       kernel_size=(1, 3),
                                                       stride=[1, 1],
                                                       padding=[0, 1]))

    def downsample(self, in_channels, out_channels, kernel_size, stride, padding):
        convLayer = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding),
                                  nn.InstanceNorm2d(num_features=out_channels,
                                                    affine=True),
                                  nn.ReLU())
        return convLayer

    def forward(self, x):
        # x has shape [batch_size, num_features, frames]
        # discriminator requires shape [batchSize, 1, num_features, frames]
        x = x.unsqueeze(1)
        conv_layer_1 = self.convLayer1(x)
        downsample1 = self.downSample1(conv_layer_1)
        downsample2 = self.downSample2(downsample1)
        downsample3 = self.downSample3(downsample2)
        output = torch.sigmoid(self.outputConvLayer(downsample3))
        return output


if __name__ == '__main__':
    # Non exhaustive test for ICRCycleGAN-VC models

    # Generator Dimensionality Testing
    np.random.seed(0)

    residual_in_channels = 256
    # input = np.random.randn(2, 80, 64)
    input = np.random.randn(2, 80, 64)
    input = torch.from_numpy(input).float()
    # print("Generator input: ", input.shape)
    mask = torch.ones_like(input)
    generator = Generator(input.shape[1:], residual_in_channels)
    output = generator(input, mask)
    # print("Generator output shape: ", output.shape)

    # Discriminator Dimensionality Testing
    discriminator = Discriminator(input.shape[1:], residual_in_channels)
    output = discriminator(output)
    # print("Discriminator output shape ", output.shape)
