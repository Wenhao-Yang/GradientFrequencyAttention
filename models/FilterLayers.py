#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: FilterLayers.py
@Time: 2021/10/15 09:20
@Overview:
"""
import numpy as np
import torch
import torch.nn as nn
from scipy import interpolate
import gradients.constants as c

class SqueezeExcitation(nn.Module):
    # input should be like [Batch, channel, time, frequency]
    def __init__(self, inplanes, reduction_ratio=4):
        super(SqueezeExcitation, self).__init__()
        self.reduction_ratio = reduction_ratio

        self.glob_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(inplanes, max(int(inplanes / self.reduction_ratio), 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(max(int(inplanes / self.reduction_ratio), 1), inplanes)
        self.activation = nn.Sigmoid()

    def forward(self, input):
        scale = self.glob_avg(input).squeeze(dim=2).squeeze(dim=2)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.activation(scale).unsqueeze(2).unsqueeze(2)

        output = input * scale

        return output


class CBAM(nn.Module):
    # input should be like [Batch, channel, time, frequency]
    def __init__(self, inplanes, planes, time_freq='both', pooling='avg'):
        super(CBAM, self).__init__()
        self.time_freq = time_freq
        self.activation = nn.Sigmoid()
        self.pooling = pooling

        self.cov_t = nn.Conv2d(inplanes, planes, kernel_size=(7, 1), stride=1, padding=(3, 0))
        self.avg_t = nn.AdaptiveAvgPool2d((None, 1))

        self.cov_f = nn.Conv2d(inplanes, planes, kernel_size=(1, 7), stride=1, padding=(0, 3))
        self.avg_f = nn.AdaptiveAvgPool2d((1, None))

        # if self.pooling == 'both':
        # self.max_t = nn.AdaptiveMaxPool2d((None, 1))
        # self.max_f = nn.AdaptiveMaxPool2d((None, 1))

    def forward(self, input):
        t_output = self.avg_t(input)
        # if self.pooling == 'both':
        #     t_output += self.max_t(input)

        t_output = self.cov_t(t_output)
        t_output = self.activation(t_output)
        t_output = input * t_output

        f_output = self.avg_f(input)
        # if self.pooling == 'both':
        #     f_output += self.max_f(input)

        f_output = self.cov_f(f_output)
        f_output = self.activation(f_output)
        f_output = input * f_output

        output = (t_output + f_output) / 2

        return output


class Mean_Norm(nn.Module):
    def __init__(self, dim=-2):
        super(Mean_Norm, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x - torch.mean(x, dim=self.dim, keepdim=True)

    def __repr__(self):
        return "Mean_Norm(dim=%d)" % self.dim


class AttentionweightLayer(nn.Module):
    def __init__(self, input_dim=161, weight='mel'):
        super(AttentionweightLayer, self).__init__()
        self.input_dim = input_dim
        self.weight = weight

        if weight == 'mel':
            m = np.arange(0, 2840.0230467083188)
            m = 700 * (10 ** (m / 2595.0) - 1)
            n = np.array([m[i] - m[i - 1] for i in range(1, len(m))])
            n = 1 / n
            x = np.arange(input_dim) * 8000 / (input_dim - 1)  # [0-8000]
            f = interpolate.interp1d(m[1:], n)
            xnew = np.arange(np.min(m[1:]), np.max(m[1:]), (np.max(m[1:]) - np.min(m[1:])) / input_dim)
            ynew = f(xnew)
            # ynew = 1 / ynew  # .max()
        elif weight == 'clean':
            ynew = c.VOX1_CLEAN
        elif weight == 'aug':
            ynew = c.VOX1_AUG
        elif weight == 'vox2':
            ynew = c.VOX2_CLEAN
        else:
            raise ValueError(weight)
        ynew = np.array(ynew)
        ynew /= ynew.max()
        self.w = nn.Parameter(torch.tensor(2.0))
        self.b = nn.Parameter(torch.tensor(-1.0))

        self.drop_p = ynew  # * dropout_p
        # self.activation = nn.Tanh()
        # self.activation = nn.Softmax(dim=-1)
        self.activation = nn.Sigmoid()

    def forward(self, x):

        assert len(self.drop_p) == x.shape[-1], print(len(self.drop_p), x.shape)

        drop_weight = torch.tensor(self.drop_p).reshape(1, 1, 1, -1).float()
        if x.is_cuda:
            drop_weight = drop_weight.cuda()

        drop_weight = self.w * drop_weight + self.b
        drop_weight = self.activation(drop_weight)

        return x * drop_weight

    def __repr__(self):
        return "AttentionweightLayer(weight=%s, w=%f, b=%f)" % (self.weight, float(self.w), float(self.b))
