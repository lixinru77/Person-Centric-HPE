import torch
import torch.nn as nn
import torch.nn.functional as F

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

# class MNet(nn.Module):
#     #将速度轴上的numFrames=8进行特征提取和压缩
#     def __init__(self, in_channels, out_channels, numFrames):
#         super(MNet, self).__init__()
#         sizeTemp = sizeTempStride = numFrames//2#8/2=4
#         #维度是[32batch*8numframe.-1,chip(numframe),range,azimuth)=(256,-1,8,64,64),卷积核（2，1，1）对应（depth,weight,width)=（8，64，64）,depth/2也就是对速度周进行8*2=4
#         self.temporalConvWx1x1 = nn.Conv3d(in_channels, out_channels, (2, 1, 1), (2, 1, 1), (0, 0, 0))
#         #将上面的depth=4//4=1
#         self.temporalMaxpool = nn.MaxPool3d((sizeTemp, 1, 1), (sizeTempStride, 1, 1))
#     def forward(self, chirpMaps):
#         chirpMaps = self.temporalConvWx1x1(chirpMaps)
#         maps = self.temporalMaxpool(chirpMaps)
#         return maps


class MNet(nn.Module):
    def __init__(self, in_channels, out_channels, numFrames):
        super(MNet, self).__init__()
        # 注意：虽然传入的 in_channels 是 2 (Real, Imag)
        # 但我们会在 forward 里把它变成 1 (Magnitude)
        # 所以后面的网络层全部按 1 通道来初始化

        # 1. 归一化层 (处理 Magnitude 的极大值)
        self.input_bn = nn.BatchNorm3d(1)

        # 2. 空间特征提取
        self.feature_extract = nn.Sequential(
            nn.Conv3d(1, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 3. 你的创新点：多普勒线性融合
        self.fusion_conv = nn.Conv3d(out_channels, out_channels, kernel_size=(numFrames, 1, 1), bias=False)
        self.bn_final = nn.BatchNorm3d(out_channels)
        self.relu_final = nn.ReLU(inplace=True)

    def forward(self, x):
        # x 的输入 shape: (Batch, 2, 8, 64, 64) -> 对应 Real 和 Imag

        real = x[:, 0:1, :, :, :]  # 取出实部 (Batch, 1, 8, 64, 64)
        imag = x[:, 1:2, :, :, :]  # 取出虚部 (Batch, 1, 8, 64, 64)

        # 模长 = sqrt(real^2 + imag^2)。加 1e-8 防止 NaN
        mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)

        # 现在的 mag shape 是 (Batch, 1, 8, 64, 64)
        # 它代表了纯粹的能量，全是正数，再怎么做卷积都不会相位抵消了！
        # ==========================================

        # 送入网络
        out = self.input_bn(mag)  # 压制异常大值
        out = self.feature_extract(out)  # 提取空间轮廓
        out = self.fusion_conv(out)  # 线性融合多普勒
        out = self.relu_final(self.bn_final(out))

        return out