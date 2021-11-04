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
import torch
import random


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


class ScriptTrainDataset(data.Dataset):
    def __init__(self, dir, samples_per_speaker, transform, num_valid=5, feat_type='kaldi',
                 loader=np.load, return_uid=False, domain=False, rand_test=False,
                 segment_len=600, verbose=1):
        self.return_uid = return_uid
        self.domain = domain
        self.rand_test = rand_test
        self.segment_len = segment_len

        feat_scp = dir + '/feats.scp' if feat_type != 'wav' else dir + '/wav.scp'
        spk2utt = dir + '/spk2utt'
        utt2spk = dir + '/utt2spk'
        utt2num_frames = dir + '/utt2num_frames'
        utt2dom = dir + '/utt2dom'

        assert os.path.exists(feat_scp), feat_scp
        assert os.path.exists(spk2utt), spk2utt

        invalid_uid = []
        base_utts = []
        total_frames = 0
        if os.path.exists(utt2num_frames):
            with open(utt2num_frames, 'r') as f:
                for l in f.readlines():
                    uid, num_frames = l.split()
                    num_frames = int(num_frames)
                    total_frames += num_frames
                    this_numofseg = int(np.ceil(float(num_frames) / segment_len))

                    for i in range(this_numofseg):
                        end = min((i + 1) * segment_len, num_frames)
                        start = min(end - segment_len, 0)
                        base_utts.append((uid, start, end))
            if verbose > 0:
                print('    There are {} basic segments.'.format(len(base_utts)))

                # if int(num_frames) < 50:
                #     invalid_uid.append(uid)
        self.base_utts = base_utts
        dataset = {}
        with open(spk2utt, 'r') as u:
            all_cls = u.readlines()
            for line in all_cls:
                spk_utt = line.split()
                spk_name = spk_utt[0]
                if spk_name not in dataset:
                    dataset[spk_name] = [x for x in spk_utt[1:] if x not in invalid_uid]

        utt2spk_dict = {}
        with open(utt2spk, 'r') as u:
            all_cls = u.readlines()
            for line in all_cls:
                utt_spk = line.split()
                uid = utt_spk[0]
                if uid in invalid_uid:
                    continue
                if uid not in utt2spk_dict:
                    utt2spk_dict[uid] = utt_spk[-1]

        self.dom_to_idx = None
        self.utt2dom_dict = None
        dom_to_idx = None
        if self.domain:
            assert os.path.exists(utt2dom), utt2dom

            utt2dom_dict = {}
            with open(utt2dom, 'r') as u:
                all_cls = u.readlines()
                for line in all_cls:
                    utt_dom = line.split()
                    uid = utt_dom[0]
                    if uid in invalid_uid:
                        continue
                    if uid not in utt2dom_dict:
                        utt2dom_dict[uid] = utt_dom[-1]

            all_domains = [utt2dom_dict[u] for u in utt2dom_dict.keys()]
            domains = list(set(all_domains))
            domains.sort()
            dom_to_idx = {domains[i]: i for i in range(len(domains))}
            self.dom_to_idx = dom_to_idx
            if verbose > 1:
                print("Domain idx: ", str(self.dom_to_idx))
            self.utt2dom_dict = utt2dom_dict

        # pdb.set_trace()

        speakers = [spk for spk in dataset.keys()]
        speakers.sort()
        if verbose > 0:
            print('==> There are {} speakers in Dataset.'.format(len(speakers)))
        spk_to_idx = {speakers[i]: i for i in range(len(speakers))}
        idx_to_spk = {i: speakers[i] for i in range(len(speakers))}

        uid2feat = {}  # 'Eric_McCormack-Y-qKARMSO7k-0001.wav': feature[frame_length, feat_dim]
        with open(feat_scp, 'r') as f:
            for line in f.readlines():
                uid, feat_offset = line.split()
                if uid in invalid_uid:
                    continue
                uid2feat[uid] = feat_offset

        if verbose > 0:
            print('    There are {} utterances in Train Dataset, where {} utterances are removed.'.format(len(uid2feat),
                                                                                                          len(invalid_uid)))
        self.valid_set = None
        self.valid_uid2feat = None
        self.valid_utt2spk_dict = None
        self.valid_utt2dom_dict = None

        if num_valid > 0:
            valid_set = {}
            valid_uid2feat = {}
            valid_utt2spk_dict = {}
            valid_utt2dom_dict = {}

            for spk in speakers:
                if spk not in valid_set.keys():
                    valid_set[spk] = []
                    for i in range(num_valid):
                        if len(dataset[spk]) <= 1:
                            break
                        j = np.random.randint(len(dataset[spk]))
                        utt = dataset[spk].pop(j)
                        valid_set[spk].append(utt)

                        valid_uid2feat[valid_set[spk][-1]] = uid2feat.pop(valid_set[spk][-1])
                        valid_utt2spk_dict[utt] = utt2spk_dict[utt]
                        if self.domain:
                            valid_utt2dom_dict[utt] = utt2dom_dict[utt]

            if verbose > 0:
                print('    Spliting {} utterances for Validation.'.format(len(valid_uid2feat)))
            self.valid_set = valid_set
            self.valid_uid2feat = valid_uid2feat
            self.valid_utt2spk_dict = valid_utt2spk_dict
            self.valid_utt2dom_dict = valid_utt2dom_dict

        self.speakers = speakers
        self.utt2spk_dict = utt2spk_dict
        self.dataset = dataset
        self.uid2feat = uid2feat
        self.spk_to_idx = spk_to_idx
        self.idx_to_spk = idx_to_spk
        self.num_spks = len(speakers)
        self.num_doms = len(self.dom_to_idx) if dom_to_idx != None else 0

        self.loader = loader
        self.feat_dim = loader(uid2feat[list(uid2feat.keys())[0]]).shape[-1]
        self.transform = transform
        if samples_per_speaker == 0:
            samples_per_speaker = np.power(2, np.ceil(np.log2(total_frames * 2 / segment_len / self.num_spks)))
            if verbose > 1:
                print(
                    '    The number of sampling utterances for each speakers is decided by the number of total frames.')
        else:
            samples_per_speaker = max(len(base_utts) / len(speakers), samples_per_speaker)
            if verbose > 1:
                print('    The number of sampling utterances for each speakers is add to the number of total frames.')

        self.samples_per_speaker = int(samples_per_speaker)
        self.c_axis = 0 if feat_type != 'wav' else 1
        self.feat_shape = (0, self.feat_dim) if feat_type != 'wav' else (1, 0)
        if verbose > 0:
            print('    Sample {} random utterances for each speakers.'.format(self.samples_per_speaker))

        if self.return_uid or self.domain:
            self.utt_dataset = []
            for i in range(self.samples_per_speaker * self.num_spks):
                sid = i % self.num_spks
                spk = self.idx_to_spk[sid]
                utts = self.dataset[spk]
                uid = utts[random.randrange(0, len(utts))]
                self.utt_dataset.append([uid, sid])

    def __getitem__(self, sid):
        # start_time = time.time()
        if self.return_uid or self.domain:
            uid, label = self.utt_dataset[sid]
            y = self.loader(self.uid2feat[uid])
            feature = self.transform(y)

            if self.domain:
                label_b = self.dom_to_idx[self.utt2dom_dict[uid]]
                return feature, label, label_b
            else:
                return feature, label, uid

        if sid < len(self.base_utts):
            while True:
                (uid, start, end) = self.base_utts[sid]
                if uid not in self.valid_utt2spk_dict:
                    y = self.loader(self.uid2feat[uid])[start:end]
                    sid = self.utt2spk_dict[uid]
                    sid = self.spk_to_idx[sid]
                    break
                else:
                    self.base_utts.pop(sid)
        else:
            rand_idxs = [sid]
            sid %= self.num_spks
            spk = self.idx_to_spk[sid]
            utts = self.dataset[spk]
            num_utt = len(utts)

            y = np.array([[]]).reshape(0, self.feat_dim)
            rand_utt_idx = np.random.randint(0, num_utt)
            rand_idxs.append(rand_utt_idx)
            uid = utts[rand_utt_idx]

            feature = self.loader(self.uid2feat[uid])
            y = np.concatenate((y, feature), axis=0)

            while len(y) < self.segment_len:
                rand_utt_idx = np.random.randint(0, num_utt)
                rand_idxs.append(rand_utt_idx)

                uid = utts[rand_utt_idx]

                feature = self.loader(self.uid2feat[uid])
                y = np.concatenate((y, feature), axis=0)

                # transform features if required
                if self.rand_test:
                    while len(rand_idxs) < 4:
                        rand_idxs.append(-1)
                    start, length = self.transform(y)
                    rand_idxs.append(start)
                    rand_idxs.append(length)

                    # [uttid uttid -1 -1 start lenght]
                    return torch.tensor(rand_idxs).reshape(1, -1), sid

        feature = self.transform(y)
        label = sid

        return feature, label

    def __len__(self):
        return self.samples_per_speaker * len(self.speakers)  # 返回一个epoch的采样数


class ScriptValidDataset(data.Dataset):
    def __init__(self, valid_set, spk_to_idx, valid_uid2feat, valid_utt2spk_dict,
                 transform, dom_to_idx=None, valid_utt2dom_dict=None, loader=np.load,
                 return_uid=False, domain=False, verbose=1):
        speakers = [spk for spk in valid_set.keys()]
        speakers.sort()

        self.dom_to_idx = dom_to_idx
        self.utt2dom_dict = valid_utt2dom_dict

        self.speakers = speakers
        self.dataset = valid_set
        self.valid_set = valid_set
        self.uid2feat = valid_uid2feat
        self.domain = domain

        uids = list(valid_uid2feat.keys())
        uids.sort()
        if verbose > 0:
            print('Examples uids: ', uids[:5])

        self.uids = uids
        self.utt2spk_dict = valid_utt2spk_dict
        self.spk_to_idx = spk_to_idx
        self.num_spks = len(speakers)

        self.loader = loader
        self.transform = transform
        self.return_uid = return_uid

    def __getitem__(self, index):
        uid = self.uids[index]
        spk = self.utt2spk_dict[uid]
        y = self.loader(self.uid2feat[uid])

        feature = self.transform(y)
        label = self.spk_to_idx[spk]

        if self.domain:
            return feature, label, self.dom_to_idx[self.utt2dom_dict[uid]]

        if self.return_uid:
            return feature, label, uid

        return feature, label

    def __len__(self):
        return len(self.uids)


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
