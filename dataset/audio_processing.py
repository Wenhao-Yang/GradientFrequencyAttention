#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: audio_processing.py
@Time: 2021/10/15 09:42
@Overview:
"""
import numpy as np
import torch
import random


NUM_FRAMES_SPECT = 300
NUM_SHIFT_SPECT = 300


class totensor(object):
    """Rescales the input PIL.Image to the given 'size'.
    If 'size' is a 2-element tuple or list in the order of (width, height), it will be the exactly size to scale.
    If 'size' is a number, it will indicate the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the exactly size or the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __call__(self, input):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        input = torch.tensor(input, dtype=torch.float32)
        return input.unsqueeze(0)


class ConcateVarInput(object):
    """Rescales the input PIL.Image to the given 'size'.
    If 'size' is a 2-element tuple or list in the order of (width, height), it will be the exactly size to scale.
    If 'size' is a number, it will indicate the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the exactly size or the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, num_frames=NUM_FRAMES_SPECT, frame_shift=NUM_SHIFT_SPECT,
                 feat_type='kaldi', remove_vad=False):

        super(ConcateVarInput, self).__init__()
        self.num_frames = num_frames
        self.remove_vad = remove_vad
        self.frame_shift = frame_shift
        self.c_axis = 0 if feat_type != 'wav' else 1

    def __call__(self, frames_features):

        network_inputs = []
        output = frames_features
        while output.shape[self.c_axis] < self.num_frames:
            output = np.concatenate((output, frames_features), axis=self.c_axis)

        input_this_file = int(np.ceil(output.shape[self.c_axis] / self.frame_shift))

        for i in range(input_this_file):
            start = i * self.frame_shift

            if start < output.shape[self.c_axis] - self.num_frames:
                end = start + self.num_frames
            else:
                start = output.shape[self.c_axis] - self.num_frames
                end = output.shape[self.c_axis]
            if self.c_axis == 0:
                network_inputs.append(output[start:end])
            else:
                network_inputs.append(output[:, start:end])

        network_inputs = torch.tensor(network_inputs, dtype=torch.float32)
        if self.remove_vad:
            network_inputs = network_inputs[:, :, 1:]

        return network_inputs


class ConcateOrgInput(object):
    """
    prepare feats with true length.
    """

    def __init__(self, remove_vad=False):
        super(ConcateOrgInput, self).__init__()
        self.remove_vad = remove_vad

    def __call__(self, frames_features):
        # pdb.set_trace()
        network_inputs = []
        output = np.array(frames_features)

        if self.remove_vad:
            output = output[:, 1:]

        network_inputs.append(output)
        network_inputs = torch.tensor(network_inputs, dtype=torch.float32)

        return network_inputs


class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0, min_chunk_size=200, max_chunk_size=400, normlize=True,
                 num_batch=0,
                 fix_len=False):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.num_batch = num_batch
        self.fix_len = fix_len
        self.normlize = normlize

        if self.fix_len:
            self.frame_len = np.random.randint(low=self.min_chunk_size, high=self.max_chunk_size)
        else:
            assert num_batch > 0
            batch_len = np.arange(self.min_chunk_size, self.max_chunk_size+1)

            print('==> Generating %d different random length...' % (len(batch_len)))

            self.batch_len = np.array(batch_len)
            print('==> Average of utterance length is %d. ' % (np.mean(self.batch_len)))

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)
        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # pdb.set_trace()
        if self.fix_len:
            frame_len = self.frame_len
        else:
            # frame_len = np.random.randint(low=self.min_chunk_size, high=self.max_chunk_size)
            frame_len = random.choice(self.batch_len)

        # pad according to max_len
        # print()
        xs = torch.stack(list(map(lambda x: x[0], batch)), dim=0)

        if frame_len < batch[0][0].shape[-2]:
            start = np.random.randint(low=0, high=batch[0][0].shape[-2] - frame_len)
            end = start + frame_len
            xs = xs[:, :, start:end, :].contiguous()
        else:
            xs = xs.contiguous()

        ys = torch.LongTensor(list(map(lambda x: x[1], batch)))

        # map_batch = map(lambda x_y: (pad_tensor(x_y[0], pad=frame_len, dim=self.dim - 1), x_y[1]), batch)
        # pad_batch = list(map_batch)
        #
        # xs = torch.stack(list(map(lambda x: x[0], pad_batch)), dim=0)
        # ys = torch.LongTensor(list(map(lambda x: x[1], pad_batch)))

        return xs, ys

    def __call__(self, batch):
        return self.pad_collate(batch)


