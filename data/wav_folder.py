"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data
import random
from data import audio
import os
import os.path
import librosa
import numpy as np

IMG_EXTENSIONS = ['.wav', '.mp3']


def is_wav_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    wavs = []
    assert os.path.isdir(dir) or os.path.islink(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            if is_wav_file(fname):
                path = os.path.join(root, fname)
                wavs.append(path)
    return wavs[:min(max_dataset_size, len(wavs))]

def read_wav(wav_path, sr=16000, mono=True):
    try:
        y, sr = librosa.load(wav_path, sr=sr, mono=True)
    except:
        raise AssertionError("Unsupported file: %s" % wav_path)
    return y, sr

def read_wavs_list(wav_dir):
    wavs = list()
    for wav in wav_dir:
        _wav, _ = read_wav(wav)
        wavs.append(_wav)
    return wavs

def write_wav(y, wav_path, sr=16000):
    librosa.output.write_wav(path=wav_path, y=y, sr=sr)
    return True

def trun_spec(y, tlen):
    start = random.choice((range(0, y.shape[0]-tlen)))
    y = y[start:start+tlen, :]
    return y
    
# def logmelfilterbank(audio,
#                      sampling_rate,
#                      fft_size=1024,
#                      hop_size=256,
#                      win_length=None,
#                      window="hann",
#                      num_mels=80,
#                      fmin=None,
#                      fmax=None,
#                      eps=1e-10,
#                      ):
#     """Compute log-Mel filterbank feature.

#     Args:
#         audio (ndarray): Audio signal (T,).
#         sampling_rate (int): Sampling rate.
#         fft_size (int): FFT size.
#         hop_size (int): Hop size.
#         win_length (int): Window length. If set to None, it will be the same as fft_size.
#         window (str): Window function type.
#         num_mels (int): Number of mel basis.
#         fmin (int): Minimum frequency in mel basis calculation.
#         fmax (int): Maximum frequency in mel basis calculation.
#         eps (float): Epsilon value to avoid inf in log calculation.

#     Returns:
#         ndarray: Log Mel filterbank feature (#frames, num_mels).

#     """
#     # get amplitude spectrogram
#     x_stft = librosa.stft(audio, n_fft=fft_size, hop_length=hop_size,
#                           win_length=win_length, window=window, pad_mode="reflect")
#     spc = np.abs(x_stft).T  # (#frames, #bins)

#     # get mel basis
#     fmin = 0 if fmin is None else fmin
#     fmax = sampling_rate / 2 if fmax is None else fmax
#     mel_basis = librosa.filters.mel(sampling_rate, fft_size, num_mels, fmin, fmax)

#     return np.log10(np.maximum(eps, np.dot(spc, mel_basis.T)))

def process_utterance(wav_path,
                      fft_size=1024,
                      hop_size=256,
                      win_length=1024,
                      window="hann",
                      num_mels=80,
                      fmin=80,
                      fmax=7600,
                      eps=1e-10,
                      sample_rate=24000,
                      loud_norm=False,
                      min_level_db=-100,
                      return_linear=False,
                      trim_long_sil=False, vocoder='pwg',
                      change_loud=False,
                      loud_range_min=0.9, loud_range_max=1.1):
    if isinstance(wav_path, str):
        if trim_long_sil:
            wav, _ = trim_long_silences(wav_path, sample_rate)
        else:
            wav, _ = librosa.core.load(wav_path, sr=sample_rate)
    else:
        wav = wav_path

    if change_loud:
        sample_num = wav.shape[0] // (sample_rate * 2) + 1  # sample point every 1 seconds
        random_point = np.random.permutation(wav.shape[0])
        sample_up, sample_down = random_point[:sample_num], random_point[sample_num:2 * sample_num]
        fp_up = np.random.uniform(2, loud_range_max, sample_num)
        fp_down = np.random.uniform(loud_range_min, 0.5, sample_num)
        fp = np.concatenate([fp_up, fp_down])
        xp = np.concatenate([sample_up, sample_down])
        index = np.argsort(xp)
        xp = xp[index]
        fp = fp[index]
        # print(xp.shape)
        change_curve = np.interp(np.arange(wav.shape[0]), xp, fp)

        wav = wav * change_curve
        if (np.abs(wav) > 1.0).sum() / wav.shape[0] > 1 / 200:
            print("too much wav out of 1", wav_path)
        wav = np.clip(wav, -1.0, 1.0)

    if loud_norm:
        assert not change_loud
        meter = pyln.Meter(sample_rate)  # create BS.1770 meter
        loudness = meter.integrated_loudness(wav)
        wav = pyln.normalize.loudness(wav, loudness, -22.0)
        if np.abs(wav).max() > 1:
            wav = wav / np.abs(wav).max()

    # get amplitude spectrogram
    x_stft = librosa.stft(wav, n_fft=fft_size, hop_length=hop_size,
                          win_length=win_length, window=window, pad_mode="constant")
    spc = np.abs(x_stft)  # (n_bins, T)

    # get mel basis
    fmin = 0 if fmin is -1 else fmin
    fmax = sample_rate / 2 if fmax is -1 else fmax
    mel_basis = librosa.filters.mel(sample_rate, fft_size, num_mels, fmin, fmax)
    mel = mel_basis @ spc

    if vocoder == 'pwg':
        mel = np.log10(np.maximum(eps, mel))  # (n_mel_bins, T)
    elif vocoder == 'waveglow':
        mel = audio.dynamic_range_compression(mel)
    else:
        assert False, f'"{vocoder}" is not in ["pwg", "waveglow"].'

    l_pad, r_pad = audio.librosa_pad_lr(wav, fft_size, hop_size, 1)
    wav = np.pad(wav, (l_pad, r_pad), mode='constant', constant_values=0.0)
    wav = wav[:mel.shape[1] * hop_size]

    if not return_linear:
        return wav, mel
    else:
        spc = audio.amp_to_db(spc)
        spc = audio.normalize(spc, {'min_level_db': min_level_db})
        return wav, mel, spc

# def default_loader(path):
#     return Image.open(path).convert('RGB')


# class ImageFolder(data.Dataset):

#     def __init__(self, root, transform=None, return_paths=False,
#                  loader=default_loader):
#         imgs = make_dataset(root)
#         if len(imgs) == 0:
#             raise(RuntimeError("Found 0 images in: " + root + "\n"
#                                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

#         self.root = root
#         self.imgs = imgs
#         self.transform = transform
#         self.return_paths = return_paths
#         self.loader = loader

#     def __getitem__(self, index):
#         path = self.imgs[index]
#         img = self.loader(path)
#         if self.transform is not None:
#             img = self.transform(img)
#         if self.return_paths:
#             return img, path
#         else:
#             return img

#     def __len__(self):
#         return len(self.imgs)
