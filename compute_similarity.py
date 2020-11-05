from data.wav_folder import read_wav
import os
from resemblyzer import preprocess_wav, VoiceEncoder 
import numpy as np

cvc_dir = '/path/to/cvc_utt_path/'
cyclegan_dir = '/path/to/cyclegan_utt_path/'
target_dir = '/path/to/tgt_utt_path/'

sampling_rate = 24000
encoder = VoiceEncoder()

cvc_list, cyclegan_list = [], []
for cvc_path, cyclegan_path in zip(os.listdir(cvc_dir), os.listdir(cyclegan_dir)):
    cvc_path = os.path.join(cvc_dir, cvc_path)
    cyclegan_path = os.path.join(cyclegan_dir, cyclegan_path)

    cvc_wav, _ = read_wav(cvc_path, sr=sampling_rate)
    cyclegan_wav, _ = read_wav(cyclegan_path, sr=sampling_rate)

    cvc_list.append(cvc_wav)
    cyclegan_list.append(cyclegan_wav)

target_list = []
for target_path in os.listdir(target_dir):
    target_path = os.path.join(target_dir, target_path)
    target_wav, _ = read_wav(target_path, sr=sampling_rate)
    target_list.append(target_wav)

spk_embeds_cvc = np.array([encoder.embed_speaker(cvc_list)])
spk_embeds_cyclegan = np.array([encoder.embed_speaker(cyclegan_list)])
spk_embeds_target = np.array([encoder.embed_speaker(target_list)])

spk_sim_cvc = np.inner(spk_embeds_cvc, spk_embeds_target)
spk_sim_cyclegan = np.inner(spk_embeds_cyclegan, spk_embeds_target)

print('CVC:{} CycleGAN:{}'.format(spk_sim_cvc, spk_sim_cyclegan))


