#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: common_func.py
@Time: 2021/10/15 09:45
@Overview:
"""

import os
import errno
import pdb
import sys
import kaldi_io
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel.distributed import DistributedDataParallel
from tqdm import tqdm
import gradients.constants as c
from loss.softmax import AdditiveMarginLinear

from models.ResNets import ResCNN
from models.TDNNs import TDNN
from evals.eval_metrics import evaluate_kaldi_eer, evaluate_kaldi_mindcf
import argparse

def create_optimizer(parameters, optimizer, **kwargs):
    # setup optimizer
    # parameters = filter(lambda p: p.requires_grad, parameters)
    if optimizer == 'sgd':
        opt = optim.SGD(parameters,
                        lr=kwargs['lr'],
                        momentum=kwargs['momentum'],
                        dampening=kwargs['dampening'],
                        weight_decay=kwargs['weight_decay'])

    elif optimizer == 'adam':
        opt = optim.Adam(parameters,
                               lr=kwargs['lr'],
                               weight_decay=kwargs['weight_decay'])

    elif optimizer == 'adagrad':
        opt = optim.Adagrad(parameters,
                            lr=kwargs['lr'],
                            lr_decay=kwargs['lr_decay'],
                            weight_decay=kwargs['weight_decay'])
    elif optimizer == 'RMSprop':
        opt = optim.RMSprop(parameters,
                            lr=kwargs['lr'],
                            momentum=kwargs['momentum'],
                            weight_decay=kwargs['weight_decay'])

    return opt

__factory = {
    'ResCNN': ResCNN,
    'TDNN': TDNN,
}

def create_model(name, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))

    model = __factory[name](**kwargs)

    if kwargs['loss_type'] in ['arcsoft']:
        model.classifier = AdditiveMarginLinear(feat_dim=kwargs['embedding_size'],
                                                num_classes=kwargs['num_classes'])

    return model


def verification_extract(extract_loader, model, xvector_dir, epoch, test_input='fix', ark_num=50000, gpu=True,
                         verbose=False, xvector=False):
    """

    :param extract_loader:
    :param model:
    :param xvector_dir:
    :param epoch:
    :param test_input:
    :param ark_num:
    :param gpu:
    :param verbose:
    :param xvector: extract xvectors in embedding-a layer
    :return:
    """
    model.eval()

    if not os.path.exists(xvector_dir):
        os.makedirs(xvector_dir)
    # pbar =
    pbar = tqdm(extract_loader, ncols=100) if verbose else extract_loader

    uid2vectors = {}
    with torch.no_grad():
        if test_input == 'fix':
            data = torch.tensor([])
            num_seg_tensor = [0]
            uid_lst = []

            batch_size = 128 if torch.cuda.is_available() else 80
            for batch_idx, (a_data, a_uid) in enumerate(pbar):
                vec_shape = a_data.shape
                if vec_shape[1] != 1:
                    a_data = a_data.reshape(vec_shape[0] * vec_shape[1], 1, vec_shape[2], vec_shape[3])

                data = torch.cat((data, a_data), dim=0)
                num_seg_tensor.append(num_seg_tensor[-1] + len(a_data))
                uid_lst.append(a_uid[0])

                if data.shape[0] >= batch_size or batch_idx + 1 == len(extract_loader):
                    if data.shape[0] > (3 * batch_size):
                        i = 0
                        out = []
                        while i < data.shape[0]:
                            data_part = data[i:(i + batch_size)]
                            data_part = data_part.cuda() if next(model.parameters()).is_cuda else data_part
                            model_out = model.xvector(data_part) if xvector else model(data_part)
                            try:
                                _, out_part, _, _ = model_out
                            except:
                                _, out_part = model_out
                            out.append(out_part)
                            i += batch_size
                        out = torch.cat(out, dim=0)
                    else:

                        data = data.cuda() if next(model.parameters()).is_cuda else data
                        model_out = model.xvector(data) if xvector else model(data)
                        try:
                            _, out, _, _ = model_out
                        except:
                            _, out = model_out

                    out = out.data.cpu().float().numpy()
                    # print(out.shape)
                    if len(out.shape) == 3:
                        out = out.squeeze(0)

                    for i, uid in enumerate(uid_lst):
                        uid2vectors[uid] = out[num_seg_tensor[i]:num_seg_tensor[i + 1]].mean(axis=0)  # , uid[0])

                    data = torch.tensor([])
                    num_seg_tensor = [0]
                    uid_lst = []

        elif test_input == 'var':
            for batch_idx, (a_data, a_uid) in enumerate(pbar):
                vec_shape = a_data.shape

                if vec_shape[1] != 1:
                    a_data = a_data.reshape(vec_shape[0] * vec_shape[1], 1, vec_shape[2], vec_shape[3])

                a_data = a_data.cuda() if next(model.parameters()).is_cuda else a_data
                if vec_shape[2] > 10 * c.NUM_FRAMES_SPECT:
                    num_half = int(vec_shape[2] / 2)
                    half_a = a_data[:, :, :num_half, :]
                    half_b = a_data[:, :, -num_half:, :]
                    a_data = torch.cat((half_a, half_b), dim=0)

                try:
                    if xvector:
                        model_out = model.module.xvector(a_data) if isinstance(model,
                                                                               DistributedDataParallel) else model.xvector(
                            a_data)
                    else:
                        model_out = model(a_data)
                except Exception as e:
                    pdb.set_trace()
                    print('\ninput shape is ', a_data.shape)
                    raise e

                try:
                    _, out, _, _ = model_out
                except:
                    _, out = model_out
                if out.shape[0] != 1:
                    out = out.mean(dim=0, keepdim=True)
                out = out.data.cpu().float().numpy()
                # print(out.shape)

                if len(out.shape) == 3:
                    out = out.squeeze(0)

                uid2vectors[a_uid[0]] = out[0]

    uids = list(uid2vectors.keys())
    # print('There are %d vectors' % len(uids))
    scp_file = xvector_dir + '/xvectors.scp'
    scp = open(scp_file, 'w')

    # write scp and ark file
    # pdb.set_trace()
    for set_id in range(int(np.ceil(len(uids) / ark_num))):
        ark_file = xvector_dir + '/xvector.{}.ark'.format(set_id)
        with open(ark_file, 'wb') as ark:
            ranges = np.arange(len(uids))[int(set_id * ark_num):int((set_id + 1) * ark_num)]
            for i in ranges:
                key = uids[i]
                vec = uid2vectors[key]
                len_vec = len(vec.tobytes())

                kaldi_io.write_vec_flt(ark, vec, key=key)
                # print(ark.tell())
                scp.write(str(uids[i]) + ' ' + str(ark_file) + ':' + str(ark.tell() - len_vec - 10) + '\n')
    scp.close()
    # print('Saving %d xvectors to %s' % (len(uids), xvector_dir))
    torch.cuda.empty_cache()

def verification_test(test_loader, dist_type, log_interval, xvector_dir, epoch):
    # switch to evaluate mode
    labels, distances = [], []
    dist_fn = nn.CosineSimilarity(dim=1).cuda() if dist_type == 'cos' else nn.PairwiseDistance(2)

    # pbar = tqdm(enumerate(test_loader))
    with torch.no_grad():
        for batch_idx, (data_a, data_p, label) in enumerate(test_loader):
            out_a = torch.tensor(data_a).cuda()  # .view(-1, 4, embedding_size)
            out_p = torch.tensor(data_p).cuda()  # .view(-1, 4, embedding_size)
            dists = dist_fn.forward(out_a, out_p).cpu().numpy()

            distances.append(dists)
            labels.append(label.numpy())
            del out_a, out_p  # , ae, pe

        # if batch_idx % log_interval == 0:
        #     pbar.set_description('Verification Epoch {}: [{}/{} ({:.0f}%)]'.format(
        #         epoch, batch_idx * len(data_a), len(test_loader.dataset), 100. * batch_idx / len(test_loader)))

    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])
    # this_xvector_dir = "%s/epoch_%s" % (xvector_dir, epoch)
    with open('%s/scores' % xvector_dir, 'w') as f:
        for d, l in zip(distances, labels):
            f.write(str(d) + ' ' + str(l) + '\n')

    eer, eer_threshold, accuracy = evaluate_kaldi_eer(distances, labels,
                                                      cos=True if dist_type == 'cos' else False,
                                                      re_thre=True)
    mindcf_01, mindcf_001 = evaluate_kaldi_mindcf(distances, labels)

    return eer, eer_threshold, mindcf_01, mindcf_001


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class NewLogger(object):
    """
    Write console output to external text file.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """

    def __init__(self, fpath=None, mode='a'):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, mode)

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
