import math
import torch
import yaml
import argparse
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from resnet3d import *
from layers import conv1x1x1, conv3x3x3

from models.networks import MultiScaleCrossSelfAttentionPRGCN

class obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, obj(b) if isinstance(b, dict) else b)

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


class FPN3D(nn.Module):
    backbones = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101,
        'resnet152': resnet152
    }

    def __init__(self, backbone='resnet34', num_classes=3, pretrained=False):
        super(FPN3D, self).__init__()
        self.numFilters = 32
        self.numGroupFrames = 8

        self.backbone_net = FPN3D.backbones[backbone](pretrained=pretrained)

        self.feature_pyramid = FeaturePyramid(self.backbone_net)

        self.l1temporalMerge = nn.Conv3d(self.numFilters * 2 * 4, self.numFilters * 2, (self.numGroupFrames //2, 1, 1), 1, 0,
                                         bias=False)
        self.l2temporalMerge = nn.Conv3d(self.numFilters * 4 * 2, self.numFilters * 4, (self.numGroupFrames // 4, 1, 1), 1,
                                         0, bias=False)
        self.temporalMerge = nn.Conv3d(self.numFilters * 8, self.numFilters * 8, (self.numGroupFrames // 8, 1, 1), 1, 0,
                                       bias=False)


    def forward(self, x):
        l1maps,l2maps,l3maps = self.feature_pyramid(x)


        maps_1 = self.l1temporalMerge(l1maps).squeeze(2)
        maps_2 = self.l2temporalMerge(l2maps).squeeze(2)
        maps_3 = self.temporalMerge(l3maps).squeeze(2)

        return maps_1,maps_2,maps_3


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('--dir', type=str, default='mscsa_prgcn', metavar='B',
                        help='directory of saving/loading')
    parser.add_argument('--visDir', type=str, default='', metavar='B',
                        help='directory of visualization')
    parser.add_argument('--config', type=str, default='mscsa_prgcn.yaml', metavar='B',
                        help='directory of visualization')
    parser.add_argument('--gpuIDs', default=[0], type=eval, help='IDs of GPUs to use')
    parser.add_argument('--pretrained', type=bool,default=True, help='pretrained model')

    parser.add_argument('--eval', action="store_true")
    parser.add_argument('-sr', '--sampling_ratio', type=int, default=1, help='sampling ratio for training/test (default: 1)')
    parser.add_argument('--keypoints', action='store_true', help='print out the APs of all keypoints')
    args = parser.parse_args()

    with open('/home/jzy/MyCode/mmPHV/config/' + args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        cfg = obj(cfg)

    net = FPN3D()
    x1 = Variable(torch.rand(11, 32, 8, 64, 64))
    x2 = Variable(torch.rand(11, 32, 8, 64, 64))
    x1_l1maps, x1_l2maps, x1_l3maps = net(x1)
    x2_l1maps, x2_l2maps, x2_l3maps = net(x2)


    # radarDecoder = MultiScaleCrossSelfAttentionPRGCN(cfg, batchnorm=False, activation=nn.PReLU)
    # output, gcn_heatmap = radarDecoder(x1_l1maps, x1_l2maps, x1_l3maps, x2_l1maps, x2_l2maps, x2_l3maps)
    # print(output,gcn_heatmap)
