# -*- coding: utf-8 -*-
# @Time    : 6/19/21 12:23 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : dataloader.py

# modified from:
# Author: David Harwath
# with some functions borrowed from https://github.com/SeanNaren/deepspeech.pytorch

# =================================================================================
# further modified by Nuo Chen (2025)

import os
import torchaudio
import numpy as np

import torch
from torch.utils.data import Dataset


def preemphasis(signal,coeff=0.97):
    """perform preemphasis on the input signal.
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])


class ICBHIDataset(Dataset):
    def __init__(self, dataset_root, audio_config, noise, epoch):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.data_root = dataset_root
        self.current_epoch = epoch
        
        data_ids = []
        for label in os.listdir(self.data_root):
            cls_datas = os.listdir(os.path.join(self.data_root, label))
            label_indices = np.ones(len(cls_datas), dtype=np.int64) * int(label)
            data_ids.extend(zip(cls_datas, label_indices))
        self.data = data_ids
        
        self.audio_conf = audio_config
        self.mode = self.audio_conf.get('mode')
        self.classes = self.audio_conf.get('num_classes')
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm')
        self.timem = self.audio_conf.get('timem')
        print('now using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        self.dataset = self.audio_conf.get('dataset')
        print('now process ' + self.dataset)
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = self.audio_conf.get('skip_norm') if self.audio_conf.get('skip_norm') else False
        if self.skip_norm:
            print('now skip normalization (use it ONLY when you are computing the normalization stats).')
        else:
            print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))
        # if add noise for data augmentation
        self.noise = noise
        if self.noise == True:
            print('now use noise augmentation')

    def _wav2fbank(self, filename):
        waveform, sr = torchaudio.load(filename)
        waveform = waveform - waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)

        target_length = self.audio_conf.get('target_length')
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        return fbank

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        datum = self.data[index]
        fbank = self._wav2fbank(os.path.join(self.data_root, str(datum[1]), datum[0]))

        # SpecAug, not do for eval set
        if self.mode == 'train':
            fbank = torch.transpose(fbank, 0, 1)
            # this is just to satisfy new torchaudio version.
            fbank = fbank.unsqueeze(0)
            if self.freqm != 0:
                freqm = torchaudio.transforms.FrequencyMasking(self.freqm//2)
                fbank = freqm(fbank)
            if self.timem != 0:
                timem = torchaudio.transforms.TimeMasking(self.timem//2)
                fbank = timem(fbank)
            # squeeze back
            fbank = fbank.squeeze(0)
            fbank = torch.transpose(fbank, 0, 1)

        # normalize the input for both training and test
        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        # skip normalization the input if you are trying to get the normalization stats.
        else:
            pass

        if self.mode == 'train' and self.noise == True:
            torch.manual_seed(2345 + self.current_epoch + index)
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)

        # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        return  fbank, torch.from_numpy(np.array(datum[1])), index

    def __len__(self):
        return len(self.data)
    