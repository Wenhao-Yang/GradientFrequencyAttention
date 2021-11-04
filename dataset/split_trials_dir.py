#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: split_trials_dir.py
@Time: 2020/11/13 11:07
@Overview:
"""
from __future__ import print_function
import argparse
import os
import shutil
import numpy as np
from kaldi_io import read_mat
from tqdm import tqdm
from kaldiio import WriteHelper
import soundfile as sf

def read_Waveform(filename):
    """
    read features from npy files
    :param filename: the path of wav files.
    :return:
    """
    # audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    # audio = audio.flatten()
    audio, sample_rate = sf.read(filename, dtype='int16')

    return audio.astype(np.float32).reshape(1, -1)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Conver flac to wav in sitw!')
    parser.add_argument('--nj', type=int, default=16, metavar='E',
                        help='number of jobs to make feats (default: 10)')
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--out-dir', type=str)
    parser.add_argument('--trials', type=str)
    parser.add_argument('--feat-format', type=str, default='kaldi')

    args = parser.parse_args()

    nj = args.nj
    data_dir = args.data_dir
    out_dir = args.out_dir
    assert os.path.exists(data_dir)
    if not os.path.exists(out_dir):
        print("Create dir for %s.." % out_dir)
        os.makedirs(out_dir)

    feats_scp_name = 'feats.scp'
    if args.feat_format == 'kaldi':
        file_loader = read_mat
    elif args.feat_format == 'npy':
        file_loader = np.load
    elif args.feat_format == 'wav':
        file_loader = read_Waveform
        feats_scp_name = 'wav.scp'

    feat_scp_path = os.path.join(data_dir, feats_scp_name)
    trials_scp = os.path.join(data_dir, args.trials)

    assert os.path.exists(feat_scp_path)
    assert os.path.exists(trials_scp)

    trials_utts = set()
    with open(trials_scp, 'r') as u:
        all_cls = u.readlines()
        for line in all_cls:
            utt_a, utt_b, target = line.split(' ')
            trials_utts.add(utt_a)
            trials_utts.add(utt_b)

    tmp_uid2feat = {}
    with open(feat_scp_path, 'r') as u:
        all_cls = tqdm(u.readlines())
        for line in all_cls:
            utt_path = line.split(' ')
            uid = utt_path[0]
            if uid in trials_utts:
                tmp_uid2feat[uid] = utt_path[-1]

    out_feat_scp = os.path.abspath(os.path.join(out_dir, 'feats.scp'))
    out_feat_ark = os.path.abspath(os.path.join(out_dir, 'feat.ark'))
    out_trials = os.path.join(out_dir, args.trials)

    writer = WriteHelper('ark,scp:%s,%s' % (out_feat_ark, out_feat_scp), compression_method=1)

    for uid in tqdm(trials_utts):
        this_feat = file_loader(tmp_uid2feat[uid])
        writer(str(uid), this_feat)

    writer.close()
    shutil.copyfile(trials_scp, out_trials)




