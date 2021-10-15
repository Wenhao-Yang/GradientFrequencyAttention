#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: ResNets.py
@Time: 2021/10/15 09:17
@Overview:
"""
import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck
from models.FilterLayers import SqueezeExcitation, CBAM, AttentionweightLayer, Mean_Norm
from models.poolings import StatisticPooling


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, reduction_ratio=8):
        super(SEBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.reduction_ratio = reduction_ratio

        # Squeeze-and-Excitation
        self.se_layer = SqueezeExcitation(inplanes=planes, reduction_ratio=reduction_ratio)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.se_layer(out)

        out += identity
        out = self.relu(out)

        return out


class MyBottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, **kwargs):
        super(MyBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class CBAMBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, reduction_ratio=16):
        super(CBAMBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.reduction_ratio = reduction_ratio

        # Squeeze-and-Excitation
        self.CBAM_layer = CBAM(planes, planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.CBAM_layer(out)

        out += identity
        out = self.relu(out)

        return out


class ResCNN(nn.Module):
    """
    Define the ResNet model with A-softmax and AM-softmax loss.
    Added dropout as https://github.com/nagadomi/kaggle-cifar10-torch7 after average pooling and fc layer.
    """

    def __init__(self, embedding_size, num_classes, input_dim=161, init_weight='mel',
                 block_type='basic', input_len=300, relu_type='relu', groups=1,
                 resnet_size=8, channels=[64, 128, 256], dropout_p=0., encoder_type='None',
                 input_norm=None, alpha=12, stride=2, transform=False, time_dim=1, fast=False,
                 avg_size=4, kernal_size=5, padding=2, filter=None, mask='None', mask_len=25,
                 gain_layer=False, **kwargs):

        super(ResCNN, self).__init__()
        resnet_type = {8: [1, 1, 1, 0],
                       10: [1, 1, 1, 1],
                       14: [2, 2, 2, 0],
                       18: [2, 2, 2, 2],
                       34: [3, 4, 6, 3],
                       50: [3, 4, 6, 3],
                       101: [3, 4, 23, 3]}

        layers = resnet_type[resnet_size]
        freq_dim = avg_size

        if block_type == "seblock":
            block = SEBasicBlock
        elif block_type == 'cbam':
            block = CBAMBlock
        elif block_type == 'bottle':
            block = MyBottleneck
        else:
            block = BasicBlock

        self.alpha = alpha
        self.layers = layers
        self.input_len = input_len
        self.input_dim = input_dim
        self.gain_layer = gain_layer

        self.dropout_p = dropout_p
        self.transform = transform
        self.fast = fast
        self.mask = mask
        self.relu_type = relu_type
        self.embedding_size = embedding_size
        self.groups = groups
        #
        if self.relu_type == 'relu6':
            self.relu = nn.ReLU6(inplace=True)
        elif self.relu_type == 'leakyrelu':
            self.relu = nn.LeakyReLU()
        else:
            self.relu = nn.ReLU(inplace=True)

        self.input_norm = input_norm
        self.input_len = input_len
        self.filter = filter

        if self.filter == 'Avg':
            self.filter_layer = nn.AvgPool2d(kernel_size=(1, 5), stride=(1, 2))
        else:
            self.filter_layer = None

        if input_norm == 'Mean':
            self.inst_layer = Mean_Norm()
        else:
            self.inst_layer = None

        if self.mask == 'attention':
            self.mask_layer = AttentionweightLayer(input_dim=input_dim, weight=init_weight)
        else:
            self.mask_layer = None

        self.inplanes = channels[0]
        self.conv1 = nn.Conv2d(1, channels[0], kernel_size=kernal_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(channels[0])
        if self.fast.startswith('avp'):
            # self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            # self.maxpool = nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.maxpool = nn.Sequential(
                nn.Conv2d(channels[0], channels[0], kernel_size=1, stride=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[0]),
                nn.AvgPool2d(kernel_size=3, stride=2)
            )
        else:
            self.maxpool = None

        # self.maxpool = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.layer1 = self._make_layer(block, channels[0], layers[0])

        self.inplanes = channels[1]
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=(5, 5), stride=2,
                               padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(channels[1])
        self.layer2 = self._make_layer(block, channels[1], layers[1])

        self.inplanes = channels[2]
        self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size=(5, 5), stride=2,
                               padding=padding, bias=False)
        self.bn3 = nn.BatchNorm2d(channels[2])
        self.layer3 = self._make_layer(block, channels[2], layers[2])

        if layers[3] != 0:
            assert len(channels) == 4
            self.inplanes = channels[3]
            stride = 1 if self.fast else 2
            self.conv4 = nn.Conv2d(channels[2], channels[3], kernel_size=(5, 5), stride=stride,
                                   padding=padding, bias=False)
            self.bn4 = nn.BatchNorm2d(channels[3])
            self.layer4 = self._make_layer(block=block, planes=channels[3], blocks=layers[3])

        self.dropout = nn.Dropout(self.dropout_p)

        last_conv_chn = channels[-1]
        freq_dim = avg_size

        if encoder_type == 'STAP':
            self.avgpool = nn.AdaptiveAvgPool2d((None, freq_dim))
            self.encoder = StatisticPooling(input_dim=last_conv_chn * freq_dim)
            self.encoder_output = last_conv_chn * freq_dim * 2
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((time_dim, freq_dim))
            self.encoder = None
            self.encoder_output = last_conv_chn * freq_dim * time_dim

        self.fc1 = nn.Sequential(
            nn.Linear(self.encoder_output, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )


        # self.fc = nn.Linear(self.inplanes * avg_size, embedding_size)
        self.classifier = nn.Linear(self.embedding_size, num_classes)

        for m in self.modules():  # 对于各层参数的初始化
            if isinstance(m, nn.Conv2d):  # 以2/n的开方为标准差，做均值为0的正态分布
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):  # weight设置为1，bias为0
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.filter_layer != None:
            x = self.filter_layer(x)

        if self.inst_layer != None:
            x = self.inst_layer(x)

        if self.mask_layer != None:
            x = self.mask_layer(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.maxpool != None:
            x = self.maxpool(x)
        x = self.layer1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.layer3(x)

        if self.layers[3] != 0:
            x = self.conv4(x)
            x = self.bn4(x)
            x = self.relu(x)
            x = self.layer4(x)

        if self.dropout_p > 0:
            x = self.dropout(x)

        x = self.avgpool(x)
        if self.encoder != None:
            x = self.encoder(x)

        x = x.view(x.size(0), -1)
        # x = self.fc(x)
        x = self.fc1(x)

        logits = self.classifier(x)

        return logits, x

    def xvector(self, x):
        if self.filter_layer != None:
            x = self.filter_layer(x)

        if self.inst_layer != None:
            x = self.inst_layer(x)

        if self.mask_layer != None:
            x = self.mask_layer(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.fast:
            x = self.maxpool(x)
        x = self.layer1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.layer3(x)

        if self.layers[3] != 0:
            x = self.conv4(x)
            x = self.bn4(x)
            x = self.relu(x)
            x = self.layer4(x)

        if self.dropout_p > 0:
            x = self.dropout(x)

        x = self.avgpool(x)
        if self.encoder != None:
            x = self.encoder(x)

        x = x.view(x.size(0), -1)
        # x = self.fc1(x)
        embeddings = self.fc1[0](x)

        return "", embeddings

