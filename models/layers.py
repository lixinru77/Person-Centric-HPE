import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gcn_networks import PRGCN
import torchvision.models as models
from models.fpn3d.resnet3d import *
from models.fpn3d.layers import conv1x1x1,conv3x3x3
from torch.autograd import Variable



class BasicBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, batchnorm=True,
                 activation=nn.ReLU):
        super(BasicBlock2D, self).__init__()
        if batchnorm:
            self.main = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                activation(),
                nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.main = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                activation(),
                nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False),
            )
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            )
        self.relu = activation()
    
    def forward(self, x):
        residual = self.downsample(x)
        out = self.main(x) + residual
        out = self.relu(out)
        return out

class BasicBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, batchnorm=True, activation=nn.ReLU):
        super(BasicBlock3D, self).__init__()
        if batchnorm:
            self.main = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm3d(out_channels),
                activation(),
                nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm3d(out_channels),
            )
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.main = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                activation(),
                nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding, bias=False),
            )
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=False),
            )
        self.relu = activation()
    
    def forward(self, x):
        residual = self.downsample(x)
        out = self.main(x) + residual
        out = self.relu(out)
        return out

class FeaturePyramid(nn.Module):
    def __init__(self, resnet):
        super(FeaturePyramid, self).__init__()

        self.resnet = resnet

        # applied in a pyramid
        self.pyramid_transformation_1 = conv3x3x3(64, 256, padding=1)
        self.pyramid_transformation_2 = conv1x1x1(256, 256)
        self.pyramid_transformation_3 = conv1x1x1(512, 256)
        self.pyramid_transformation_4 = conv1x1x1(1024, 256)
        self.pyramid_transformation_5 = conv1x1x1(2048, 256)

        # both based around resnet_feature_5
        # self.pyramid_transformation_6 = conv3x3x3(2048, 256, padding=1, stride=2)
        # self.pyramid_transformation_7 = conv3x3x3(256, 256, padding=1, stride=2)

        # applied after upsampling
        self.upsample_transform_1 = conv3x3x3(256, 256, padding=1)
        self.upsample_transform_2 = conv3x3x3(256, 256, padding=1)
        self.upsample_transform_3 = conv3x3x3(256, 256, padding=1)
        self.upsample_transform_4 = conv3x3x3(256, 256, padding=1)

    def _upsample(self, original_feature, scaled_feature, scale_factor=2):
        # is this correct? You do lose information on the upscale...
        depth, height, width = scaled_feature.size()[2:]
        return F.upsample(original_feature, scale_factor=scale_factor)[:, :, :depth, :height, :width]

    def forward(self, x):
        resnet_feature_1, resnet_feature_2, resnet_feature_3, resnet_feature_4, resnet_feature_5 = self.resnet(x)

        # pyramid_feature_6 = self.pyramid_transformation_6(resnet_feature_5)
        # pyramid_feature_7 = self.pyramid_transformation_7(F.relu(pyramid_feature_6))

        pyramid_feature_5 = self.pyramid_transformation_5(resnet_feature_5)  # transform c5 from 2048d to 256d
        pyramid_feature_4 = self.pyramid_transformation_4(resnet_feature_4)  # transform c4 from 1024d to 256d
        upsampled_feature_5 = self._upsample(pyramid_feature_5, pyramid_feature_4)  # deconv c5 to c4.size

        pyramid_feature_4 = self.upsample_transform_4(
            torch.add(upsampled_feature_5, pyramid_feature_4)  # add up-c5 and c4, and conv
        )

        pyramid_feature_3 = self.pyramid_transformation_3(resnet_feature_3)  # transform c3 from 512d to 256d
        upsampled_feature_4 = self._upsample(pyramid_feature_4, pyramid_feature_3)  # deconv c4 to c3.size

        pyramid_feature_3 = self.upsample_transform_3(
            torch.add(upsampled_feature_4, pyramid_feature_3)  # add up-c4 and c3, and conv
        )

        pyramid_feature_2 = self.pyramid_transformation_2(resnet_feature_2)  # c2 is 256d, so no need to transform
        upsampled_feature_3 = self._upsample(pyramid_feature_3, pyramid_feature_2)  # deconv c3 to c2.size

        pyramid_feature_2 = self.upsample_transform_2(
            torch.add(upsampled_feature_3, pyramid_feature_2)  # add up-c3 and c2, and conv
        )

        pyramid_feature_1 = self.pyramid_transformation_1(resnet_feature_1)  # use conv3x3x3 up c1 from 64d to 256d
        upsampled_feature_2 = self._upsample(pyramid_feature_2, pyramid_feature_1)  # deconv c2 to c1.size

        pyramid_feature_1 = self.upsample_transform_1(
            torch.add(upsampled_feature_2, pyramid_feature_1)  # add up-c2 and c1, and conv
        )

        return (pyramid_feature_1,  # 8
                pyramid_feature_2,  # 16
                pyramid_feature_3)  # 32


class SimpleFeaturePyramid(nn.Module):
    #########xinru
    # def __init__(self, resnet):
    def __init__(self, resnet,backbone_name='resnet50'):
        super(SimpleFeaturePyramid, self).__init__()

        self.resnet = resnet

        #########xinru
        in_ch = 256 if backbone_name == 'resnet18' else 1024


        self.deconv1 = nn.ConvTranspose3d(in_channels=in_ch, out_channels=256,
                                          kernel_size=(1, 3, 3),
                                          stride=(1, 2, 2),  # 步长s，将输入空间的每个维度分别放大2倍
                                          padding=(0, 1, 1),  # 填充p，确保输出的空间维度为输入的两倍
                                          output_padding=(0, 1, 1)
                                          )

        self.deconv2 = nn.ConvTranspose3d(in_channels=in_ch,
                                          out_channels=256,
                                          kernel_size=(2, 5, 5),  # 调整卷积核大小
                                          stride=(1, 5, 5),  # 调整步长
                                          padding=(0, 3, 3),  # 调整填充
                                          output_padding=(0, 2, 2)
                                          )

        self.deconv3 = nn.ConvTranspose3d(in_channels=in_ch,
                                          out_channels=256,
                                          kernel_size=(4, 9, 9),  # 调整卷积核大小
                                          stride=(1, 9, 9),  # 调整步长
                                          padding=(0, 5, 5),  # 调整填充
                                          output_padding=(0, 6, 6)
                                          )


    def forward(self, x):
        _, _, _, resnet_feature_4, _ = self.resnet(x)

        pyramid_feature_3 = self.deconv1(resnet_feature_4)
        pyramid_feature_2 = self.deconv2(resnet_feature_4)
        pyramid_feature_1 = self.deconv3(resnet_feature_4) # 32,256,4,32,32


        return (pyramid_feature_1,  # 32,256,4,32,32
                pyramid_feature_2,  # 32,256,2,16,16
                pyramid_feature_3)  # 32,256,1,8,8

class MultiScaleCrossSelfAttentionPRGCN(nn.Module):
    def __init__(self, cfg, batchnorm=True, activation=nn.ReLU):
        super(MultiScaleCrossSelfAttentionPRGCN, self).__init__()
        self.numGroupFrames = cfg.DATASET.numGroupFrames
        self.numFilters = cfg.MODEL.numFilters
        self.width = cfg.DATASET.heatmapSize
        self.height = cfg.DATASET.heatmapSize
        self.numKeypoints = cfg.DATASET.numKeypoints
        self.numObjs = cfg.DATASET.numObjs

        self.decoderLayer3 = nn.Sequential(
            BasicBlock2D(self.numFilters*8*4, self.numFilters*8, 3, 1, 1, batchnorm, activation),
            # BasicBlock2D(self.numFilters*8*2, self.numFilters*8, 3, 1, 1, batchnorm, activation),
            BasicBlock2D(self.numFilters*8, self.numFilters*4, 3, 1, 1, batchnorm, activation),
            nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True),
        )
        self.decoderLayer2 = nn.Sequential(
            BasicBlock2D(self.numFilters*4*5, self.numFilters*4, 3, 1, 1, batchnorm, activation),
            # BasicBlock2D(self.numFilters*4*3, self.numFilters*4, 3, 1, 1, batchnorm, activation),
            BasicBlock2D(self.numFilters*4, self.numFilters*2, 3, 1 ,1, batchnorm, activation),
            nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True),
        )
        self.decoderLayer1 = nn.Sequential(
            BasicBlock2D(self.numFilters*2*5, self.numFilters*2, 3, 1, 1, batchnorm, activation),
            # BasicBlock2D(self.numFilters*2*3, self.numFilters*2, 3, 1, 1, batchnorm, activation),
            BasicBlock2D(self.numFilters*2, self.numFilters, 3, 1, 1, batchnorm, activation),
            nn.Conv2d(self.numFilters, self.numKeypoints * self.numObjs, 1, 1, 0, bias=False),
        )

        A = torch.tensor([
            [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],#RHip
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],#RKnee
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],#RAnkle
            [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],#LHip
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],#LKnee
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],#LAnkle
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],#Neck
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],#Head
            [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0],#LShoulder
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],#LElbow
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],#LWrist
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0],#RShoulder
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],#RElbow
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]#RWrist
        ], dtype=torch.float).cuda()
        self.gcn = PRGCN(cfg, A)

        filterList = [self.numFilters*8, self.numFilters*4, self.numFilters*2]
        self.phi_cross_hori = nn.ModuleList([nn.Conv2d(i, i, 1, 1, 0, bias=False) for i in filterList])
        self.theta_cross_hori = nn.ModuleList([nn.Conv2d(i, i, 1, 1, 0, bias=False) for i in filterList])
        self.phi_cross_vert = nn.ModuleList([nn.Conv2d(i, i, 1, 1, 0, bias=False) for i in filterList])
        self.theta_cross_vert = nn.ModuleList([nn.Conv2d(i, i, 1, 1, 0, bias=False) for i in filterList])
        self.phi_self_hori = nn.ModuleList([nn.Conv2d(i, i, 1, 1, 0, bias=False) for i in filterList])
        self.theta_self_hori = nn.ModuleList([nn.Conv2d(i, i, 1, 1, 0, bias=False) for i in filterList])
        self.phi_self_vert = nn.ModuleList([nn.Conv2d(i, i, 1, 1, 0, bias=False) for i in filterList])
        self.theta_self_vert = nn.ModuleList([nn.Conv2d(i, i, 1, 1, 0, bias=False) for i in filterList])
        self.sigmoid = nn.Sigmoid()

    def attention(self, k, q, maps):
        b, c, h, w  = maps.size()
        k, q = k.view(b, c, h * w), q.view(b, c, h * w)
        spat_attn = torch.einsum('bij,bik->bjk', (k, q))
        maps = maps.view(b, c, h * w)
        maps = torch.einsum('bci,bik->bck', (maps, F.softmax(spat_attn, 1).to(maps.dtype)))
        maps = maps.view(b, c, h, w)
        return maps

    def forward(self, ral1maps, ral2maps, ramaps, rel1maps, rel2maps, remaps):
        ramaps_res = ramaps
        remaps_res = remaps

        # TODO：注释这里
        k3_c_hori = self.phi_cross_hori[0](ramaps)
        q3_c_vert = self.theta_cross_vert[0](remaps)

        k3_c_vert = self.phi_cross_vert[0](remaps)
        q3_c_hori = self.theta_cross_hori[0](ramaps)

        k3_hori = self.phi_self_hori[0](ramaps)
        q3_hori = self.theta_self_hori[0](ramaps)

        k3_vert = self.phi_self_vert[0](remaps)
        q3_vert = self.theta_self_vert[0](remaps)

        ramaps_cross = self.attention(k3_c_hori, q3_c_vert, ramaps) + ramaps_res
        ramaps_self = self.attention(k3_hori, q3_hori, ramaps)
        remaps_cross = self.attention(k3_c_vert, q3_c_hori, remaps) + remaps_res
        remaps_self = self.attention(k3_vert, q3_vert, remaps)
        maps = self.decoderLayer3(torch.cat((ramaps_cross, ramaps_self, remaps_cross, remaps_self), 1))

        # maps = self.decoderLayer3(torch.cat((ramaps_res, remaps_res), 1))

        ral2maps_res = ral2maps
        rel2maps_res = rel2maps
        k2_c_hori = self.phi_cross_hori[1](ral2maps)
        q2_c_vert = self.theta_cross_vert[1](rel2maps)
        k2_c_vert = self.phi_cross_vert[1](rel2maps)
        q2_c_hori = self.theta_cross_hori[1](ral2maps)
        k2_hori = self.phi_self_hori[1](ral2maps)
        q2_hori = self.theta_self_hori[1](ral2maps)
        k2_vert = self.phi_self_vert[1](rel2maps)
        q2_vert = self.theta_self_vert[1](rel2maps)
        ral2maps_cross = self.attention(k2_c_hori, q2_c_vert, ral2maps) + ral2maps_res
        ral2maps_self = self.attention(k2_hori, q2_hori, ral2maps)
        rel2maps_cross = self.attention(k2_c_vert, q2_c_hori, rel2maps) + rel2maps_res
        rel2maps_self = self.attention(k2_vert, q2_vert, rel2maps)
        maps = self.decoderLayer2(torch.cat((maps, ral2maps_cross, ral2maps_self, rel2maps_cross, rel2maps_self), 1))
        # maps = self.decoderLayer2(torch.cat((maps, ral2maps_res, rel2maps_res), 1))

        ral1maps_res = ral1maps
        rel1maps_res = rel1maps
        k1_c_hori = self.phi_cross_hori[2](ral1maps)
        q1_c_vert = self.theta_cross_vert[2](rel1maps)
        k1_c_vert = self.phi_cross_vert[2](rel1maps)
        q1_c_hori = self.theta_cross_hori[2](ral1maps)
        k1_hori = self.phi_self_hori[2](ral1maps)
        q1_hori = self.theta_self_hori[2](ral1maps)
        k1_vert = self.phi_self_vert[2](rel1maps)
        q1_vert = self.theta_self_vert[2](rel1maps)
        ral1maps_cross = self.attention(k1_c_hori, q1_c_vert, ral1maps) + ral1maps_res
        ral1maps_self = self.attention(k1_hori, q1_hori, ral1maps)
        rel1maps_cross = self.attention(k1_c_vert, q1_c_hori, rel1maps) + rel1maps_res
        rel1maps_self = self.attention(k1_vert, q1_vert, rel1maps)
        maps = self.decoderLayer1(torch.cat((maps, ral1maps_cross, ral1maps_self, rel1maps_cross, rel1maps_self), 1))

        # maps = self.decoderLayer1(torch.cat((maps, ral1maps_res, rel1maps_res), 1))

        # TODO:HuPR注释这行，否则加这行
        maps = F.interpolate(maps, scale_factor=2.0, mode='bilinear', align_corners=True) #32,14,32,32-->32,14,64,64

        gcn_output = self.gcn(maps) # 32,1,14,32,32
        return maps, gcn_output
        # return maps


class Encoder3D(nn.Module):
    backbones = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101,
        'resnet152': resnet152
    }

    #######xinru
    # def __init__(self, cfg, batchnorm=True, activation=nn.ReLU, backbone='resnet101'):
    def __init__(self, cfg, batchnorm=True, activation=nn.ReLU, backbone='resnet50'):
        super(Encoder3D, self).__init__()
        # self.numFrames = cfg.DATASET.numFrames
        self.numGroupFrames = cfg.DATASET.numGroupFrames  # for 60
        self.numFilters = cfg.MODEL.numFilters
        self.width = cfg.DATASET.heatmapSize
        self.height = cfg.DATASET.heatmapSize

        self.backbone_net = Encoder3D.backbones[backbone](pretrained=False)
        ###############xinru
        # self.feature_pyramid = SimpleFeaturePyramid(self.backbone_net)
        self.feature_pyramid = SimpleFeaturePyramid(self.backbone_net,backbone_name=backbone)

        self.l1temporalMerge = nn.Conv3d(self.numFilters * 2 * 4, self.numFilters * 2, (self.numGroupFrames//2, 1, 1), 1, 0,
                                         bias=False)
        self.l2temporalMerge = nn.Conv3d(self.numFilters * 4 * 2, self.numFilters * 4, (self.numGroupFrames // 4, 1, 1), 1,
                                         0, bias=False)
        self.temporalMerge = nn.Conv3d(self.numFilters * 8, self.numFilters * 8, (self.numGroupFrames // 8, 1, 1), 1, 0,
                                       bias=False)

    def forward(self, maps):
        l1maps, l2maps, l3maps = self.feature_pyramid(maps)
        maps_1 = self.l1temporalMerge(l1maps).squeeze(2) # 32,64,32,32
        maps_2 = self.l2temporalMerge(l2maps).squeeze(2) # 32,128,16,16
        maps_3 = self.temporalMerge(l3maps).squeeze(2) # 32,256,8,8

        return maps_1, maps_2, maps_3


class BehaviorClassifier(nn.Module):
    def __init__(self, cfg):
        super(BehaviorClassifier, self).__init__()
        self.in_channels = cfg.DATASET.numKeypoints
        self.numBehaviors = 3
        self.heatmapSize = self.width = self.height = cfg.DATASET.heatmapSize
        self.out_channels = 512
        self.resnet18 = models.resnet18(pretrained=True)
        # self.EfficentNet = models.efficientnet_b0(pretrained=True)
        self.resnet18.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18.fc = nn.Linear(self.out_channels, self.numBehaviors)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, gcn_maps):
        gcn_maps_adj = gcn_maps.reshape(-1, self.in_channels, self.height, self.width)
        y = self.resnet18(gcn_maps_adj)
        prob = self.softmax(y)
        return prob


