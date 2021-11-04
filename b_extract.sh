#!/usr/bin/env bash

# author: yangwenhao
# contact: 874681044@qq.com
# file: b_extract.sh
# time: 2021/11/4 21:16
# Description: 

data_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification

if [ $stage -le 0 ]; then
  model=ResCNN
  dataset=vox1
  train_set=vox1
  test_set=vox1
  feat_type=klsp
  feat=log
  loss=arcsoft
  resnet_size=8
  encoder_type=None
  embedding_size=256
  block_type=cbam
  kernel=5,5
  cam=grad_cam
  echo -e "\n\033[1;4;31m stage${stage} Training ${model}_${encoder_type} in ${train_set}_${test_set} with ${loss}\033[0m\n"
  for cam in gradient ;do
    python gradients/cam_extract.py \
      --model ${model} \
      --resnet-size ${resnet_size} \
      --cam ${cam} \
      --start-epochs 50 \
      --epochs 50 \
      --train-dir ${data_dir}/data/${dataset}/${feat_type}/dev \
      --train-set-name vox1 \
      --test-set-name vox1 \
      --test-dir ${data_dir}/data/${test_set}/${feat_type}/test \
      --input-norm Mean \
      --kernel-size ${kernel} \
      --stride 2 \
      --channels 64,128,256 \
      --encoder-type ${encoder_type} \
      --block-type ${block_type} \
      --time-dim 1 \
      --avg-size 4 \
      --embedding-size ${embedding_size} \
      --alpha 0 \
      --loss-type ${loss} \
      --dropout-p 0.1 \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}/${input_norm}_${block_type}_${encoder_type}_dp125_alpha${alpha}_em${embedding_size}_${weight}_chn64_wd5e4_var \
      --extract-path Data/gradient/${model}${resnet_size}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}/${input_norm}_${block_type}_${encoder_type}_dp125_alpha${alpha}_em${embedding_size}_${weight}_chn64_wd5e4_var/epoch_50_var_${cam} \
      --gpu-id 1 \
      --margin 0.2 \
      --s 30 \
      --sample-utt 1211
  done

  for s in dev dev_aug_com;do
    python gradients/visual_gradient.py \
      --extract-path Data/gradient/${model}${resnet_size}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}/${input_norm}_${block_type}_${encoder_type}_dp125_alpha${alpha}_em${embedding_size}_${weight}_chn64_wd5e4_var/epoch_50_var_${cam} \
      --feat-dim 161 \
      --acoustic-feature spectrogram
  done
  # The gradient should be stored in 'extract_path/train.grad.npy'

  exit
fi


if [ $stage -le 10 ]; then
  model=ResCNN
  dataset=vox2
  train_set=vox2
  test_set=vox1
  feat_type=klsp
  feat=log
  loss=arcsoft
  resnet_size=8
  encoder_type=None
  embedding_size=256
  block_type=cbam
  kernel=5,5
  cam=grad_cam
  echo -e "\n\033[1;4;31m stage${stage} Training ${model}_${encoder_type} in ${train_set}_${test_set} with ${loss}\033[0m\n"
  for cam in gradient ;do
    python gradients/cam_extract.py \
      --model ${model} \
      --resnet-size ${resnet_size} \
      --cam ${cam} \
      --start-epochs 50 \
      --epochs 50 \
      --train-dir ${data_dir}/data/${dataset}/${feat_type}/dev \
      --train-set-name vox2 \
      --test-set-name vox1 \
      --test-dir ${data_dir}/data/${test_set}/${feat_type}/test \
      --input-norm Mean \
      --kernel-size ${kernel} \
      --stride 2 \
      --channels 64,128,256 \
      --encoder-type ${encoder_type} \
      --block-type ${block_type} \
      --time-dim 1 \
      --avg-size 4 \
      --embedding-size ${embedding_size} \
      --alpha 0 \
      --loss-type ${loss} \
      --dropout-p 0.1 \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_baseline/${loss}/${input_norm}_${block_type}_${encoder_type}_dp01_alpha${alpha}_em${embedding_size}_var \
      --extract-path Data/gradient/${model}${resnet_size}/${datasets}/${feat_type}_egs_baseline/${loss}/${input_norm}_${block_type}_${encoder_type}_dp01_alpha${alpha}_em${embedding_size}_var/epoch_50_var_${cam} \
      --gpu-id 1 \
      --margin 0.2 \
      --s 30 \
      --sample-utt 5994
  done

  for s in dev ;do
    python gradients/visual_gradient.py \
      --extract-path Data/gradient/${model}${resnet_size}/${datasets}/${feat_type}_egs_baseline/${loss}/${input_norm}_${block_type}_${encoder_type}_dp01_alpha${alpha}_em${embedding_size}_var/epoch_50_var_${cam} \
      --feat-dim 161 \
      --acoustic-feature spectrogram
  done
fi