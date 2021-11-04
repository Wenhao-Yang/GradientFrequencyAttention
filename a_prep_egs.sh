#!/usr/bin/env bash

# author: yangwenhao
# contact: 874681044@qq.com
# file: a_prep_egs.sh
# time: 2021/10/4 21:15
# Description:

data_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification
stage=1

# VoxCeleb1
if [ $stage -le 1 ]; then
  dataset=vox1
  feat=klsp
  feat_type=klsp
  # for each speaker, sample 1024 egs with 600 frames.
  # 2 utterances are kept for valid.

  echo -e "\n\033[1;4;31m Stage ${stage}: making ${feat} egs with kaldi spectrogram for ${dataset}\033[0m\n"
  for s in 161; do
    python dataset/make_egs.py \
      --data-dir ${data_dir}/data/${dataset}/${feat}/dev \
      --out-dir ${data_dir}/data/${dataset}/egs/${feat} \
      --nj 12 \
      --feat-type ${feat_type} \
      --train \
      --input-per-spks 1024 \
      --num-frames 600 \
      --feat-format kaldi \
      --out-format kaldi_cmp \
      --num-valid 2 \
      --out-set dev

    python dataset/make_egs.py \
      --data-dir ${data_dir}/data/${dataset}/${feat}/dev_${s} \
      --out-dir ${data_dir}/data/${dataset}/egs/${feat} \
      --nj 12 \
      --feat-type ${feat_type} \
      --num-frames 600 \
      --input-per-spks 1024 \
      --feat-format kaldi \
      --out-format kaldi_cmp \
      --num-valid 2 \
      --out-set valid
  done
  exit
fi


# VoxCeleb2
if [ $stage -le 2 ]; then
  dataset=vox2
  feat=klsp
  feat_type=klsp

  echo -e "\n\033[1;4;31m Stage ${stage}: making ${feat} egs with kaldi spectrogram for ${dataset}\033[0m\n"
  for s in 161; do
    python dataset/make_egs.py \
      --data-dir ${data_dir}/data/${dataset}/${feat}/dev \
      --out-dir ${data_dir}/data/${dataset}/egs/${feat} \
      --nj 12 \
      --feat-type ${feat_type} \
      --train \
      --input-per-spks 768 \
      --num-frames 600 \
      --feat-format kaldi \
      --out-format kaldi_cmp \
      --num-valid 2 \
      --out-set dev

    python dataset/make_egs.py \
      --data-dir ${data_dir}/data/${dataset}/${feat}/dev_${s} \
      --out-dir ${data_dir}/data/${dataset}/egs/${feat} \
      --nj 12 \
      --feat-type ${feat_type} \
      --num-frames 600 \
      --input-per-spks 768 \
      --feat-format kaldi \
      --out-format kaldi_cmp \
      --num-valid 2 \
      --out-set valid
  done
  exit
fi