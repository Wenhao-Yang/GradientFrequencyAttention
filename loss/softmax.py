#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: softmax.py
@Time: 2021/10/15 09:29
@Overview:
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class AdditiveMarginLinear(nn.Module):
    def __init__(self, feat_dim, num_classes, use_gpu=False):
        super(AdditiveMarginLinear, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.W = torch.nn.Parameter(torch.randn(feat_dim, num_classes), requires_grad=True)
        if use_gpu:
            self.W.cuda()
        nn.init.xavier_normal(self.W, gain=1)

    def forward(self, x):
        # assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.feat_dim

        # pdb.set_trace()
        # x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        # x_norm = torch.div(x, x_norm)

        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.W, dim=0)  # torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        # w_norm = torch.div(self.W, w_norm)

        costh = torch.mm(x_norm, w_norm)  # .clamp_(min=-1., max=1.)

        return costh

    def __repr__(self):
        return "AdditiveMarginLinear(feat_dim=%f, num_classes=%d)" % (self.feat_dim, self.num_classes)


class ArcSoftmaxLoss(nn.Module):

    def __init__(self, margin=0.5, s=64, iteraion=0, all_iteraion=0):
        super(ArcSoftmaxLoss, self).__init__()
        self.s = s
        self.margin = margin
        self.ce = nn.CrossEntropyLoss()
        self.iteraion = iteraion
        self.all_iteraion = all_iteraion

    def forward(self, costh, label):
        lb_view = label.view(-1, 1)
        theta = costh.acos()

        if lb_view.is_cuda:
            lb_view = lb_view.cpu()

        delt_theta = torch.zeros(costh.size()).scatter_(1, lb_view.data, self.margin)

        # pdb.set_trace()
        if costh.is_cuda:
            delt_theta = Variable(delt_theta.cuda())

        costh_m = (theta + delt_theta).cos()
        if self.iteraion < self.all_iteraion:
            costh_m = 0.5 * costh + 0.5 * costh_m
            self.iteraion += 1

        costh_m_s = self.s * costh_m
        loss = self.ce(costh_m_s, label)

        return loss

    def __repr__(self):
        return "ArcSoftmaxLoss(margin=%f, s=%d, iteration=%d, all_iteraion=%s)" % (self.margin,
                                                                                   self.s,
                                                                                   self.iteraion,
                                                                                   self.all_iteraion)
