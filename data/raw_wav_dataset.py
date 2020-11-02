from data.base_dataset import BaseDataset
from data.wav_folder import make_dataset, read_wav
import librosa
import torch
import numpy as np
import os
import random
import yaml


class WavDataset(BaseDataset):
    def __init__(self, opt, sampling_rate=16000, n_frames=128, frame_period=5.0, num_mcep=36):
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

        self.n_frames = 72000

    def trun_wav(self, wav, frames):
        if wav.shape[0] < frames:
            len_pad = frames - wav.shape[0]
            wav = np.pad(wav, ((0,len_pad)), 'constant')
        elif wav.shape[0] > frames:
            start = random.choice((range(0, wav.shape[0]-self.n_frames)))
            wav = wav[start:start+self.n_frames]
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

        A_wav, _ = read_wav(A_path, sr=24000)
        B_wav, _ = read_wav(B_path, sr=24000)

        A_wav = self.trun_wav(A_wav, self.n_frames)
        B_wav = self.trun_wav(B_wav, self.n_frames)
            
        A = torch.from_numpy(A_wav).float()
        B = torch.from_numpy(B_wav).float()

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return min(self.A_size, self.B_size)


if __name__ == '__main__':
    trainA = np.random.randn(162, 24, 554)
    trainB = np.random.randn(158, 24, 554)
    dataset = trainingDataset(trainA, trainB)
    trainLoader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=2,
                                              shuffle=True)
    for epoch in range(10):
        for i, (trainA, trainB) in enumerate(trainLoader):
            print(trainA.shape, trainB.shape)
