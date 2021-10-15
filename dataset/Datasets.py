#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: Datasets.py
@Time: 2021/10/15 09:39
@Overview:
"""
import os
from tqdm import tqdm
import numpy as np
import torch.utils.data as data
from kaldi_io import read_mat

class KaldiExtractDataset(data.Dataset):
    def __init__(self, dir, transform, filer_loader, trials_file='trials', extract_trials=True, verbose=0):

        feat_scp = dir + '/feats.scp'
        trials = dir + '/%s' % trials_file

        if os.path.exists(trials) and extract_trials:
            assert os.path.exists(feat_scp), feat_scp

            trials_utts = set()
            with open(trials, 'r') as u:
                all_cls = u.readlines()
                for line in all_cls:
                    try:
                        utt_a, utt_b, target = line.split()
                    except Exception as e:
                        print("error: [", line, "]")
                        raise e

                    trials_utts.add(utt_a)
                    trials_utts.add(utt_b)

            uid2feat = {}
            with open(feat_scp, 'r') as u:
                all_cls = tqdm(u.readlines()) if verbose > 0 else u.readlines()
                for line in all_cls:
                    utt_path = line.split(' ')
                    uid = utt_path[0]
                    if uid in trials_utts:
                        uid2feat[uid] = utt_path[-1]

        else:
            print("    trials not exist!")
            uid2feat = {}
            with open(feat_scp, 'r') as u:
                all_cls = tqdm(u.readlines())
                for line in all_cls:
                    utt_path = line.split(' ')
                    uid = utt_path[0]
                    uid2feat[uid] = utt_path[-1]

        # pdb.set_trace()
        utts = list(uid2feat.keys())
        utts.sort()
        # assert len(utts) == len(trials_utts)
        if verbose > 0:
            print('==> There are {} utterances in Verifcation set to extract vectors.'.format(len(utts)))

        self.uid2feat = uid2feat
        self.transform = transform
        self.uids = utts
        self.file_loader = filer_loader

    def __getitem__(self, index):
        uid = self.uids[index]
        y = self.file_loader(self.uid2feat[uid])
        feature = self.transform(y)

        return feature, uid

    def __len__(self):
        return len(self.uids)  # 返回一个epoch的采样数


class ScriptVerifyDataset(data.Dataset):
    def __init__(self, dir, xvectors_dir, trials_file='trials', loader=np.load, return_uid=False):

        feat_scp = xvectors_dir + '/xvectors.scp'
        trials = dir + '/%s' % trials_file

        if not os.path.exists(feat_scp):
            raise FileExistsError(feat_scp)
        if not os.path.exists(trials):
            raise FileExistsError(trials)

        uid2feat = {}
        with open(feat_scp, 'r') as f:
            for line in f.readlines():
                uid, feat_offset = line.split()
                uid2feat[uid] = feat_offset

        utts = set(uid2feat.keys())

        # print('\n==> There are {} utterances in Verification trials.'.format(len(uid2feat)))

        trials_pair = []
        positive_pairs = 0
        skip_pairs = 0
        with open(trials, 'r') as t:
            all_pairs = t.readlines()
            for line in all_pairs:
                pair = line.split()
                if pair[2] == 'nontarget' or pair[2] == '0':
                    pair_true = False
                else:
                    pair_true = True
                    positive_pairs += 1
                if pair[0] in utts and pair[1] in utts:
                    trials_pair.append((pair[0], pair[1], pair_true))
                else:
                    skip_pairs += 1

        trials_pair = np.array(trials_pair)
        # trials_pair = trials_pair[trials_pair[:, 2].argsort()[::-1]]

        # print('    There are {} pairs in trials with {} positive pairs'.format(len(trials_pair),
        #                                                                        positive_pairs))

        self.uid2feat = uid2feat
        self.trials_pair = trials_pair
        self.numofpositive = positive_pairs

        self.loader = loader
        self.return_uid = return_uid

    def __getitem__(self, index):
        uid_a, uid_b, label = self.trials_pair[index]

        feat_a = self.uid2feat[uid_a]
        feat_b = self.uid2feat[uid_b]
        data_a = self.loader(feat_a)
        data_b = self.loader(feat_b)

        if label in ['True', True]:
            label = True
        else:
            label = False

        if self.return_uid:
            # pdb.set_trace()
            # print(uid_a, uid_b)
            return data_a, data_b, label, uid_a, uid_b

        return data_a, data_b, label

    def partition(self, num):
        if num > len(self.trials_pair):
            print('%d is greater than the total number of pairs')

        else:
            self.trials_pair = self.trials_pair[:num]

        assert len(self.trials_pair) == num
        num_positive = 0
        for x, y, z in self.trials_pair:
            if z in ['True', True]:
                num_positive += 1

        assert len(self.trials_pair) == num, '%d != %d' % (len(self.trials_pair), num)
        assert self.numofpositive == num_positive, '%d != %d' % (self.numofpositive, num_positive)
        print('%d positive pairs remain.' % num_positive)

    def __len__(self):
        return len(self.trials_pair)


class EgsDataset(data.Dataset):
    def __init__(self, dir, feat_dim, transform, loader=read_mat, domain=False,
                 random_chunk=[], batch_size=0):

        feat_scp = dir + '/feats.scp'

        if not os.path.exists(feat_scp):
            raise FileExistsError(feat_scp)

        dataset = []
        spks = set([])
        doms = set([])

        with open(feat_scp, 'r') as u:
            all_cls_upath = tqdm(u.readlines())
            for line in all_cls_upath:
                try:
                    cls, upath = line.split()
                    dom_cls = -1
                except ValueError as v:
                    cls, dom_cls, upath = line.split()
                    dom_cls = int(dom_cls)

                cls = int(cls)

                dataset.append((cls, dom_cls, upath))
                doms.add(dom_cls)
                spks.add(cls)

        print('==> There are {} speakers in Dataset.'.format(len(spks)))
        print('    There are {} egs in Dataset'.format(len(dataset)))

        self.dataset = dataset
        self.feat_dim = feat_dim
        self.loader = loader
        self.transform = transform
        self.num_spks = len(spks)
        self.num_doms = len(doms)
        self.domain = domain
        self.chunk_size = []
        self.batch_size = batch_size

    def __getitem__(self, idx):
        # time_s = time.time()
        # print('Starting loading...')
        label, dom_label, upath = self.dataset[idx]

        y = self.loader(upath)

        feature = self.transform(y)
        # time_e = time.time()
        # print('Using %d for loading egs' % (time_e - time_s))
        if self.domain:
            return feature, label, dom_label
        else:
            return feature, label

    def __len__(self):
        return len(self.dataset)  # 返回一个epoch的采样数
