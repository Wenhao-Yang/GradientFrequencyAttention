#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: TDNNs.py
@Time: 2021/10/15 09:17
@Overview:
"""
import torch
import numpy as np
import torch.nn as nn

from models.FilterLayers import Mean_Norm, AttentionweightLayer
from models.poolings import StatisticPooling


def get_activation(activation):
    if activation == 'relu':
        nonlinearity = nn.ReLU
    elif activation in ['leakyrelu', 'leaky_relu']:
        nonlinearity = nn.LeakyReLU
    elif activation == 'prelu':
        nonlinearity = nn.PReLU

    return nonlinearity

class TimeDelayLayer(nn.Module):

    def __init__(self, input_dim=23, output_dim=512, context_size=5, stride=1, dilation=1,
                 dropout_p=0.0, padding=0, groups=1, activation='relu'):
        super(TimeDelayLayer, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.padding = padding
        self.groups = groups

        self.kernel = nn.Conv1d(self.input_dim, self.output_dim, self.context_size, stride=self.stride,
                                padding=self.padding, dilation=self.dilation, groups=self.groups)

        if activation == 'relu':
            self.nonlinearity = nn.ReLU(inplace=True)
        elif activation in ['leakyrelu', 'leaky_relu']:
            self.nonlinearity = nn.LeakyReLU()
        elif activation == 'prelu':
            self.nonlinearity = nn.PReLU()

        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)
        outpu: size (batch, new_seq_len, output_features)
        '''
        x = self.kernel(x.transpose(1, 2))
        x = self.nonlinearity(x)
        x = self.bn(x)

        return x.transpose(1, 2)


class TDNN(nn.Module):
    def __init__(self, num_classes, embedding_size, input_dim, alpha=0., input_norm='Mean',
                 filter=None, feat_dim=64, dropout_p=0.0, dropout_layer=False, encoder_type='STAP', activation='relu',
                 num_classes_b=0, block_type='basic', stride=[1],
                 mask='None', channels=[512, 512, 512, 512, 1500], **kwargs):
        super(TDNN, self).__init__()
        self.num_classes = num_classes
        self.num_classes_b = num_classes_b
        self.dropout_p = dropout_p
        self.dropout_layer = dropout_layer
        self.input_dim = input_dim
        self.channels = channels
        self.alpha = alpha
        self.mask = mask
        self.filter = filter
        self.feat_dim = feat_dim
        self.block_type = block_type.lower()
        self.stride = stride
        self.activation = activation

        if len(self.stride) == 1:
            while len(self.stride) < 4:
                self.stride.append(self.stride[0])
        if np.sum((self.stride)) > 4:
            print('The stride for tdnn layers are: ', str(self.stride))
        if activation != 'relu':
            print('The activation function is : ', activation)
        nonlinearity = get_activation(activation)

        self.filter_layer = None

        if input_norm == 'Instance':
            self.inst_layer = nn.InstanceNorm1d(input_dim)
        elif input_norm == 'Mean':
            self.inst_layer = Mean_Norm()
        else:
            self.inst_layer = None

        if self.mask == 'attention':
            self.mask_layer = AttentionweightLayer(input_dim=input_dim, weight=init_weight)
        else:
            self.mask_layer = None

        if self.filter_layer != None:
            self.input_dim = feat_dim

        if self.block_type == 'basic':
            TDlayer = TimeDelayLayer
        else:
            raise ValueError(self.block_type)


        self.frame1 = TimeDelayLayer(input_dim=self.input_dim, output_dim=self.channels[0],
                                        context_size=5, stride=self.stride[0], dilation=1,
                                        activation=self.activation, groups=first_groups)

        self.frame2 = TDlayer(input_dim=self.channels[0], output_dim=self.channels[1],
                              context_size=3, stride=self.stride[1], dilation=2, activation=self.activation)
        self.frame3 = TDlayer(input_dim=self.channels[1], output_dim=self.channels[2],
                              context_size=3, stride=self.stride[2], dilation=3, activation=self.activation)
        self.frame4 = TDlayer(input_dim=self.channels[2], output_dim=self.channels[3],
                              context_size=1, stride=self.stride[0], dilation=1, activation=self.activation)
        self.frame5 = TimeDelayLayer(input_dim=self.channels[3], output_dim=self.channels[4],
                                        context_size=1, stride=self.stride[3], dilation=1,
                                        activation=self.activation)

        self.drop = nn.Dropout(p=self.dropout_p)

        if encoder_type == 'STAP':
            self.encoder = StatisticPooling(input_dim=self.channels[4])
            self.encoder_output = self.channels[4] * 2
        else:
            raise ValueError(encoder_type)

        self.segment6 = nn.Sequential(
            nn.Linear(self.encoder_output, 512),
            nonlinearity(),
            nn.BatchNorm1d(512)
        )

        self.segment7 = nn.Sequential(
            nn.Linear(512, embedding_size),
            nonlinearity(),
            nn.BatchNorm1d(embedding_size)
        )

        if num_classes > 0:
            self.classifier = nn.Linear(embedding_size, num_classes)
        else:
            print("Set not classifier in xvectors model!")
            self.classifier = None
        # self.bn = nn.BatchNorm1d(num_classes)

        for m in self.modules():  # 对于各层参数的初始化
            if isinstance(m, nn.BatchNorm1d):  # weight设置为1，bias为0
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, TimeDelayLayer):
                # nn.init.normal(m.kernel.weight, mean=0., std=1.)
                nn.init.kaiming_normal_(m.kernel.weight, mode='fan_out', nonlinearity='relu')

    def set_global_dropout(self, dropout_p):
        self.dropout_p = dropout_p
        self.drop.p = dropout_p

    def forward(self, x):
        # pdb.set_trace()
        # print(x.shape)

        if len(x.shape) == 4:
            x = x.squeeze(1).float()

        if self.inst_layer != None:
            x = self.inst_layer(x)

        if self.mask_layer != None:
            x = self.mask_layer(x)

        x = self.frame1(x)
        x = self.frame2(x)
        x = self.frame3(x)
        x = self.frame4(x)
        x = self.frame5(x)

        if self.dropout_layer:
            x = self.drop(x)

        # print(x.shape)
        x = self.encoder(x)
        embedding_a = self.segment6(x)
        embedding_b = self.segment7(embedding_a)

        if self.classifier == None:
            logits = ""
        else:
            logits = self.classifier(embedding_b)

        return logits, embedding_b

    def xvector(self, x):
        # pdb.set_trace()
        if self.filter_layer != None:
            x = self.filter_layer(x)

        if len(x.shape) == 4:
            x = x.squeeze(1).float()

        if self.inst_layer != None:
            x = self.inst_layer(x)

        if self.mask_layer != None:
            x = self.mask_layer(x)

        x = self.frame1(x)
        x = self.frame2(x)
        x = self.frame3(x)
        x = self.frame4(x)
        x = self.frame5(x)

        if self.dropout_layer:
            x = self.drop(x)

        # print(x.shape)
        x = self.encoder(x)
        embedding_a = self.segment6[0](x)

        return "", embedding_a
