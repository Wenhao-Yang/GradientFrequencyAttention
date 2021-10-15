#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: poolings.py
@Time: 2021/10/15 09:27
@Overview:
"""
import torch
import torch.nn as nn


class StatisticPooling(nn.Module):

    def __init__(self, input_dim):
        super(StatisticPooling, self).__init__()
        self.input_dim = input_dim

    def forward(self, x):
        """
        :param x:   [length,feat_dim] vector
        :return:   [feat_dim] vector
        """
        x_shape = x.shape
        if len(x.shape) != 3:
            x = x.reshape(x_shape[0], x_shape[-2], -1)

        assert x.shape[-1] == self.input_dim

        mean_x = x.mean(dim=1)
        std_x = x.var(dim=1, unbiased=False).add_(1e-12).sqrt()
        # std_x = x.std(dim=1, unbiased=False)
        mean_std = torch.cat((mean_x, std_x), 1)
        return mean_std
