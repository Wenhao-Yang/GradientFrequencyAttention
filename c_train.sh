#!/bin/bash

# """
# @Author: yangwenhao
# @Contact: 874681044@qq.com
# @Software: PyCharm
# @File: c_train.sh
# @Time: 2021/10/15 09:37
# @Overview:
# """

stage=0
# stage=0 is ResCNN-64 training by VoxCeleb2 dev set.
# stage=10 is ResCNN training and stage=20 is TDNN training.

data_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification

if [ $stage -le 0 ]; then
  datasets=vox2
  feat_type=klsp
  model=ResCNN
  resnet_size=8      # The depth of the model
  encoder_type=None  # Global Average Pooling
  embedding_size=256 # Embedding size.
  block_type=cbam    # CBAM blocks.
  kernel=5,5         # The kernel size for downsample conv layers.
  alpha=0            # Embedding l2 normalization.
  input_norm=Mean    # Substract mean for input spectrograms.

  for loss in arcsoft; do
    echo -e "\n\033[1;4;31m Training ${model}${resnet_size} in ${datasets}_egs with ${loss} with ${input_norm} normalization \033[0m\n"
    python train_egs.py \
      --model ${model} \
      --train-dir ${data_dir}/data/${datasets}/egs/${feat_type}/dev \
      --train-test-dir ${data_dir}/data/vox1/${feat_type}/dev/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${data_dir}/data/${datasets}/egs/${feat_type}/dev_valid \
      --test-dir ${data_dir}/data/vox1/${feat_type}/test \
      --feat-format kaldi \
      --random-chunk 200 400 \
      --input-norm ${input_norm} \
      --resnet-size ${resnet_size} \
      --nj 12 \
      --epochs 17 \
      --scheduler rop \
      --patience 2 \
      --accu-steps 1 \
      --lr 0.1 \
      --milestones 10,20,30,40 \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_baseline/${loss}/${input_norm}_${block_type}_${encoder_type}_dp01_alpha${alpha}_em${embedding_size}_var \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_baseline/${loss}/${input_norm}_${block_type}_${encoder_type}_dp01_alpha${alpha}_em${embedding_size}_var/checkpoint_50.pth \
      --kernel-size ${kernel} \
      --channels 64,128,256 \
      --stride 2 \
      --batch-size 128 \
      --embedding-size ${embedding_size} \
      --time-dim 1 \
      --avg-size 4 \
      --encoder-type ${encoder_type} \
      --block-type ${block_type} \
      --num-valid 2 \
      --alpha ${alpha} \
      --margin 0.2 \
      --s 30 \
      --m 3 \
      --loss-ratio 0.01 \
      --weight-decay 0.0001 \
      --dropout-p 0.1 \
      --gpu-id 0,1 \
      --extract \
      --cos-sim \
      --all-iteraion 0 \
      --loss-type ${loss}
  done
  exit
fi

if [ $stage -le 10 ]; then
  datasets=vox1
  feat_type=klsp
  model=ResCNN
  resnet_size=8
  encoder_type=None
  embedding_size=256
  block_type=cbam
  kernel=5,5
  loss=arcsoft
  alpha=0
  input_norm=Mean
  mask_layer=None
  # ResCNN-64
  for weight in none; do
    echo -e "\n\033[1;4;31m Stage${stage}: Training ${model}${resnet_size} in ${datasets}_egs with ${loss} with ${input_norm} normalization \033[0m\n"
    python train_egs.py \
      --model ${model} \
      --train-dir ${data_dir}/data/${datasets}/egs/${feat_type}/dev \
      --train-test-dir ${data_dir}/data/vox1/${feat_type}/dev/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${data_dir}/data/${datasets}/egs/${feat_type}/dev_valid \
      --test-dir ${data_dir}/data/vox1/${feat_type}/test \
      --feat-format kaldi \
      --random-chunk 200 400 \
      --input-norm ${input_norm} \
      --input-dim 161 \
      --resnet-size ${resnet_size} \
      --nj 12 \
      --epochs 50 \
      --scheduler rop \
      --patience 2 \
      --accu-steps 1 \
      --lr 0.1 \
      --mask-layer ${mask_layer} \
      --milestones 10,20,30,40 \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}/${input_norm}_${block_type}_${encoder_type}_dp125_alpha${alpha}_em${embedding_size}_${weight}_chn64_wd5e4_var \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}/${input_norm}_${block_type}_${encoder_type}_dp125_alpha${alpha}_em${embedding_size}_${weight}_chn64_wd5e4_var/checkpoint_50.pth \
      --kernel-size ${kernel} \
      --channels 64,128,256 \
      --stride 2 \
      --batch-size 128 \
      --embedding-size ${embedding_size} \
      --time-dim 1 \
      --avg-size 4 \
      --encoder-type ${encoder_type} \
      --block-type ${block_type} \
      --num-valid 2 \
      --alpha ${alpha} \
      --margin 0.2 \
      --s 30 \
      --m 3 \
      --loss-ratio 0.01 \
      --weight-decay 0.0005 \
      --dropout-p 0.125 \
      --gpu-id 0,1 \
      --extract \
      --cos-sim \
      --all-iteraion 0 \
      --loss-type ${loss}
  done
  
  mask_layer=attention
  # ResCNN-32
  for weight in mel clean aug vox2; do
    echo -e "\n\033[1;4;31m Stage${stage}: Training ${model}${resnet_size} in ${datasets}_egs with ${loss} with ${input_norm} normalization \033[0m\n"
    python train_egs.py \
      --model ${model} \
      --train-dir ${data_dir}/data/${datasets}/egs/${feat_type}/dev \
      --train-test-dir ${data_dir}/data/vox1/${feat_type}/dev/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${data_dir}/data/${datasets}/egs/${feat_type}/dev_valid \
      --test-dir ${data_dir}/data/vox1/${feat_type}/test \
      --feat-format kaldi \
      --random-chunk 200 400 \
      --input-norm ${input_norm} \
      --input-dim 161 \
      --resnet-size ${resnet_size} \
      --nj 12 \
      --epochs 50 \
      --scheduler rop \
      --patience 2 \
      --accu-steps 1 \
      --lr 0.1 \
      --mask-layer ${mask_layer} \
      --init-weight ${weight} \
      --milestones 10,20,30,40 \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}/${input_norm}_${block_type}_${encoder_type}_dp125_alpha${alpha}_em${embedding_size}_${weight}_chn32_wd5e4_var \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}/${input_norm}_${block_type}_${encoder_type}_dp125_alpha${alpha}_em${embedding_size}_${weight}_chn32_wd5e4_var/checkpoint_50.pth \
      --kernel-size ${kernel} \
      --channels 32,64,128 \
      --stride 2 \
      --batch-size 128 \
      --embedding-size ${embedding_size} \
      --time-dim 1 \
      --avg-size 4 \
      --encoder-type ${encoder_type} \
      --block-type ${block_type} \
      --num-valid 2 \
      --alpha ${alpha} \
      --margin 0.2 \
      --s 30 \
      --m 3 \
      --loss-ratio 0.01 \
      --weight-decay 0.0005 \
      --dropout-p 0.125 \
      --gpu-id 0,1 \
      --extract \
      --cos-sim \
      --all-iteraion 0 \
      --loss-type ${loss}
  done

  # ResCNN-16
  for weight in mel clean aug vox2; do
    echo -e "\n\033[1;4;31m Stage${stage}: Training ${model}${resnet_size} in ${datasets}_egs with ${loss} with ${input_norm} normalization \033[0m\n"
    python train_egs.py \
      --model ${model} \
      --train-dir ${data_dir}/data/${datasets}/egs/${feat_type}/dev \
      --train-test-dir ${data_dir}/data/vox1/${feat_type}/dev/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${data_dir}/data/${datasets}/egs/${feat_type}/dev_valid \
      --test-dir ${data_dir}/data/vox1/${feat_type}/test \
      --feat-format kaldi \
      --random-chunk 200 400 \
      --input-norm ${input_norm} \
      --input-dim 161 \
      --resnet-size ${resnet_size} \
      --nj 12 \
      --epochs 50 \
      --scheduler rop \
      --patience 2 \
      --accu-steps 1 \
      --lr 0.1 \
      --mask-layer ${mask_layer} \
      --init-weight ${weight} \
      --milestones 10,20,30,40 \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}/${input_norm}_${block_type}_${encoder_type}_dp125_alpha${alpha}_em${embedding_size}_${weight}_chn16_wd5e4_var \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}/${input_norm}_${block_type}_${encoder_type}_dp125_alpha${alpha}_em${embedding_size}_${weight}_chn16_wd5e4_var/checkpoint_50.pth \
      --kernel-size ${kernel} \
      --channels 16,32,64 \
      --stride 2 \
      --batch-size 128 \
      --embedding-size ${embedding_size} \
      --time-dim 1 \
      --avg-size 4 \
      --encoder-type ${encoder_type} \
      --block-type ${block_type} \
      --num-valid 2 \
      --alpha ${alpha} \
      --margin 0.2 \
      --s 30 \
      --m 3 \
      --loss-ratio 0.01 \
      --weight-decay 0.0005 \
      --dropout-p 0.125 \
      --gpu-id 0,1 \
      --extract \
      --cos-sim \
      --all-iteraion 0 \
      --loss-type ${loss}
  done
  exit
fi




# TDNN
if [ $stage -le 20 ]; then
  model=TDNN
  datasets=vox1
  #  feat=fb24
  feat_type=klsp
  loss=soft
  encod=STAP
  embedding_size=256
  input_dim=161
  input_norm=Mean
  # _lrr${lr_ratio}_lsr${loss_ratio}
  feat=klsp

  for loss in arcsoft; do
    echo -e "\n\033[1;4;31m Stage ${stage}: Training ${model}_${encod} in ${datasets}_${feat} with ${loss}\033[0m\n"
    python train_egs.py \
      --train-dir ${data_dir}/data/${datasets}/egs/${feat_type}/dev \
      --train-test-dir ${data_dir}/data/vox1/${feat_type}/dev/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${data_dir}/data/${datasets}/egs/${feat_type}/dev_valid \
      --test-dir ${data_dir}/data/vox1/${feat_type}/test \
      --nj 12 \
      --epochs 40 \
      --patience 2 \
      --milestones 10,20,30 \
      --model ${model} \
      --scheduler rop \
      --weight-decay 0.0005 \
      --lr 0.1 \
      --alpha 0 \
      --feat-format kaldi \
      --embedding-size ${embedding_size} \
      --var-input \
      --batch-size 128 \
      --accu-steps 1 \
      --shuffle \
      --random-chunk 200 400 \
      --input-dim ${input_dim} \
      --channels 512,512,512,512,1500 \
      --encoder-type ${encod} \
      --check-path Data/checkpoint/${model}/${datasets}/${feat_type}_egs_baseline/${loss}/${input_norm}_${encod}_em${embedding_size}_wd5e4_var \
      --resume Data/checkpoint/${model}/${datasets}/${feat_type}_egs_baseline/${loss}/${input_norm}_${encod}_em${embedding_size}_wd5e4_var/checkpoint_13.pth \
      --cos-sim \
      --dropout-p 0.0 \
      --veri-pairs 9600 \
      --gpu-id 0,1 \
      --num-valid 2 \
      --loss-type ${loss} \
      --margin 0.2 \
      --s 30 \
      --log-interval 10
  done

  for loss in arcsoft; do

    echo -e "\n\033[1;4;31m Stage ${stage}: Training ${model}_${encod} in ${datasets}_${feat} with ${loss}\033[0m\n"
    python train_egs.py \
      --train-dir ${data_dir}/data/${datasets}/egs/${feat_type}/dev \
      --train-test-dir ${data_dir}/data/vox1/${feat_type}/dev/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${data_dir}/data/${datasets}/egs/${feat_type}/dev_valid \
      --test-dir ${data_dir}/data/vox1/${feat_type}/test \
      --nj 12 \
      --epochs 40 \
      --patience 2 \
      --milestones 10,20,30 \
      --model ${model} \
      --scheduler rop \
      --weight-decay 0.0005 \
      --lr 0.1 \
      --alpha 0 \
      --feat-format kaldi \
      --embedding-size ${embedding_size} \
      --var-input \
      --batch-size 128 \
      --accu-steps 1 \
      --shuffle \
      --random-chunk 200 400 \
      --input-dim ${input_dim} \
      --channels 256,256,256,256,768 \
      --encoder-type ${encod} \
      --check-path Data/checkpoint/${model}/${datasets}/${feat_type}_egs_baseline/${loss}/${input_norm}_${encod}_em${embedding_size}_chn256_wd5e4_var \
      --resume Data/checkpoint/${model}/${datasets}/${feat_type}_egs_baseline/${loss}/${input_norm}_${encod}_em${embedding_size}_chn256_wd5e4_var/checkpoint_13.pth \
      --cos-sim \
      --dropout-p 0.0 \
      --veri-pairs 9600 \
      --gpu-id 0,1 \
      --num-valid 2 \
      --loss-type ${loss} \
      --margin 0.2 \
      --s 30 \
      --log-interval 10
  done
  loss=arcsoft
  for weight in mel aug vox2; do
    mask_layer=attention
#    weight=clean

    echo -e "\n\033[1;4;31m Stage ${stage}: Training ${model}_${encod} in ${datasets}_${feat} with ${loss}\033[0m\n"
    python train_egs.py \
      --train-dir ${data_dir}/data/${datasets}/egs/${feat_type}/dev \
      --train-test-dir ${data_dir}/data/vox1/${feat_type}/dev/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${data_dir}/data/${datasets}/egs/${feat_type}/dev_valid \
      --test-dir ${data_dir}/data/vox1/${feat_type}/test \
      --nj 12 \
      --epochs 40 \
      --patience 2 \
      --milestones 10,20,30 \
      --model ${model} \
      --scheduler rop \
      --weight-decay 0.0005 \
      --lr 0.1 \
      --alpha 0 \
      --feat-format kaldi \
      --embedding-size ${embedding_size} \
      --var-input \
      --batch-size 128 \
      --accu-steps 1 \
      --shuffle \
      --random-chunk 200 400 \
      --input-dim ${input_dim} \
      --mask-layer ${mask_layer} \
      --init-weight ${weight} \
      --channels 256,256,256,256,768 \
      --encoder-type ${encod} \
      --check-path Data/checkpoint/${model}/${datasets}/${feat_type}_egs_attention/${loss}/${input_norm}_${encod}_em${embedding_size}_${weight}_wd5e4_var \
      --resume Data/checkpoint/${model}/${datasets}/${feat_type}_egs_attention/${loss}/${input_norm}_${encod}_em${embedding_size}_${weight}_wd5e4_var/checkpoint_13.pth \
      --cos-sim \
      --dropout-p 0.0 \
      --veri-pairs 9600 \
      --gpu-id 0,1 \
      --num-valid 2 \
      --loss-type ${loss} \
      --margin 0.2 \
      --s 30 \
      --log-interval 10
  done
  exit
fi