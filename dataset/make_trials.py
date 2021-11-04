#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: make_trials.py
@Time: 2020/3/29 4:14 PM
@Overview:
"""

import os
import pdb
import random
import sys
from tqdm import tqdm
import numpy as np

random.seed(123456)
np.random.seed(123456)

if sys.argv[1].isdigit():
    num_pair = int(sys.argv[1])
    data_roots = sys.argv[2:]
else:
    num_pair = -1
    data_roots = sys.argv[1:]

print('Current path: ' + os.getcwd())
assert len(data_roots)>0
print("Dirs are: " + '; '.join(data_roots))


for data_dir in data_roots:
    spk2utt = data_dir+'/spk2utt'
    assert os.path.exists(spk2utt)
    utt2spk = data_dir + '/utt2spk'
    assert os.path.exists(utt2spk)

    spk2utt_dict = {}
    with open(spk2utt, 'r') as f:
        lines = f.readlines()
        for l in lines:
            lst = l.split()
            spkid = lst[0]
            spk2utt_dict[spkid]=lst[1:]

    utt2spk_dict = {}
    with open(utt2spk, 'r') as f:
        lines = f.readlines()
        for l in lines:
            lst = l.split()
            utt = lst[0]
            utt2spk_dict[utt] = lst[1]

    trials = data_dir+'/trials'

    with open(trials, 'w') as f:
        trials = []
        utts = len(list(utt2spk_dict.keys()))
        spks = list(spk2utt_dict.keys())

        # num_repeat = int((len(spks) - 1) * 5)
        # if utts*num_repeat*len(spks)>30*num_pair:
        #     num_repeat = int(10*num_pair/len(spks))

        print('Num of repeats: %d ' % (num_pair/len(spks)))
        pairs = 0
        positive_pairs = set()
        negative_pairs = set()

        random.shuffle(spks)
        pbar = tqdm(range(len(spks)))
        for spk_idx in pbar:
            spk = spks[spk_idx]
            other_spks = spks.copy()
            other_spks.pop(spk_idx)

            num_utt= len(spk2utt_dict[spk])
            spk_posi = 0
            for i in range(num_utt):
                for j in range(i+1, num_utt):
                    if spk_posi>=int(0.7*num_pair/len(spks)):
                        break
                    this_line = ' '.join((spk2utt_dict[spk][i], spk2utt_dict[spk][j], 'target\n'))
                    this_line_r = ' '.join((spk2utt_dict[spk][j], spk2utt_dict[spk][i], 'target\n'))
                    # f.write(this_line)
                    if this_line_r not in positive_pairs:
                        positive_pairs.add(this_line)
                        spk_posi+=1

            for i in range(int(0.75*num_pair/len(spks))):
                this_uid = np.random.choice(spk2utt_dict[spk])
                other_spk = np.random.choice(other_spks)
                other_uid = np.random.choice(spk2utt_dict[other_spk])

                this_line = ' '.join((this_uid, other_uid, 'nontarget\n'))
                this_line_r = ' '.join((other_uid, this_uid, 'nontarget\n'))
                # f.write(this_line)
                if len(positive_pairs) < 10 * num_pair:
                    if this_line_r not in negative_pairs:
                        negative_pairs.add(this_line)
                else:
                    break
                # trials.append((this_line, 0))
                # pairs += 1

        positive_pairs = list(positive_pairs)
        negative_pairs = list(negative_pairs)
        # pdb.set_trace()

        random.shuffle(negative_pairs)
        random.shuffle(positive_pairs)

        if len(positive_pairs)>0.5*num_pair:
            positive_pairs=positive_pairs[:int(0.5*num_pair)]

        num_positive = len(positive_pairs)
        for l in negative_pairs:
            positive_pairs.append(l)
            if len(positive_pairs)>=num_pair:
                break

        random.shuffle(positive_pairs)
        for l in positive_pairs:
            f.write(l)

        print('Generate %d pairs for set: %s, in which %d of them are positive pairs.' % (
            num_pair, data_dir, num_positive))
