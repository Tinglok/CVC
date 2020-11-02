from data.base_dataset import BaseDataset
from data.wav_folder import make_dataset, read_wav, process_utterance
# from data.preprocess import world_encode_data, transpose_in_list, logf0_statistics, coded_sps_normalization_fit_transform
import librosa
import torch
import numpy as np
import os
import random


class WavDataset(BaseDataset):
    def __init__(self, opt, sampling_rate=24000, n_frames=48000):
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        self.n_frames = n_frames
        self.sampling_rate = sampling_rate

    def trun_spec(self, spec, frames):
        # if spec.shape[0] < frames:
        #     len_pad = frames - spec.shape[0]
        #     spec = np.pad(spec, ((0,len_pad),(0,0)), 'constant', constant_values=(0.))
        if spec.shape[0] > frames:
            start = random.choice((range(0, spec.shape[0]-frames)))
            spec = spec[start:start+frames,:]
        return spec

    def trun_wav(self, wav, frames):
        # if wav.shape[0] < frames:
        #     len_pad = frames - wav.shape[0]
        #     wav = np.pad(wav, ((0,len_pad)), 'constant', constant_values=(0.))
        if wav.shape[0] > frames:
            start = random.choice((range(0, wav.shape[0]-frames)))
            wav = wav[start:start+frames]
        return wav

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        # coded_sps_A_norm = self.coded_sps_A_norm[index % self.A_size]
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        # coded_sps_B_norm = self.coded_sps_B_norm[index_B]

        A_wav, _ = read_wav(A_path, sr=self.sampling_rate)
        B_wav, _ = read_wav(B_path, sr=self.sampling_rate)

        A_wav = self.trun_wav(A_wav, self.n_frames)
        B_wav = self.trun_wav(B_wav, self.n_frames)

        # extract mel spectrogram
        A_wav, A_mel = process_utterance(A_wav, sample_rate=self.sampling_rate)
        B_wav, B_mel = process_utterance(B_wav, sample_rate=self.sampling_rate)
        
        A = torch.from_numpy(A_mel).float()
        B = torch.from_numpy(B_mel).float()

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        # return min(self.A_size, self.B_size)
        return max(self.A_size, self.B_size)
