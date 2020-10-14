import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5,
                 pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(ConvBlock(in_planes, n1x1, kernel_size=1))

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            ConvBlock(in_planes, n3x3red, kernel_size=1),
            ConvBlock(n3x3red, n3x3, kernel_size=3, padding=1))

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            ConvBlock(in_planes, n5x5red, kernel_size=1),
            ConvBlock(n5x5red, n5x5, kernel_size=3, padding=1),
            ConvBlock(n5x5, n5x5, kernel_size=3, padding=1),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            ConvBlock(in_planes, pool_planes, kernel_size=1),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1)


class Inception_Naive(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3, n5x5):
        super(Inception_Naive, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(ConvBlock(in_planes, n1x1, kernel_size=1))

        # 3x3 conv branch
        self.b2 = nn.Sequential(
            ConvBlock(in_planes, n3x3, kernel_size=3, padding=1))

        # 5x5 conv branch
        self.b3 = nn.Sequential(
            ConvBlock(in_planes, n5x5, kernel_size=3, padding=1))

        # 3x3 pool
        self.b4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1))

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1)


class GoogLeNet(nn.Module):
    def __init__(self, naive=False):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            ConvBlock(3, 192, kernel_size=3, padding=1))
        if (naive):
            self.a3 = Inception_Naive(192, 64, 128, 32)
            self.b3 = Inception_Naive(416, 128, 192, 96)

            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

            self.a4 = Inception_Naive(832, 192, 208, 48)
            self.b4 = Inception_Naive(1280, 160, 224, 64)
            self.c4 = Inception_Naive(1728, 128, 256, 64)
            self.d4 = Inception_Naive(2176, 112, 288, 64)
            self.e4 = Inception_Naive(2640, 256, 320, 128)

            self.a5 = Inception_Naive(3344, 256, 320, 128)
            self.b5 = Inception_Naive(4048, 384, 384, 128)

            self.avgpool = nn.AvgPool2d(8, stride=1)
            self.linear = nn.Linear(4944, 100)
        else:
            self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
            self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

            self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
            self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
            self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
            self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
            self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

            self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
            self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

            self.avgpool = nn.AvgPool2d(8, stride=1)
            self.linear = nn.Linear(1024, 100)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
