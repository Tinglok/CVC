from data.wav_folder import read_wav
import os
from resemblyzer import preprocess_wav, VoiceEncoder 
import numpy as np

NCE_dir = '/com_space/zhaohang/CUT/checkpoints/voice_CUT_replicate_many_p256/converted_sound_independent_MM/'
cyclegan_dir = '/com_space/zhaohang/CUT/checkpoints/cyclegan_replicate_many_p256/converted_sound_independent_MM/'
target_dir = '/com_space/zhaohang/CUT/datasets/voice/p256/'

sampling_rate = 24000
encoder = VoiceEncoder()

NCE_list, cyclegan_list = [], []
for NCE_path, cyclegan_path in zip(os.listdir(NCE_dir), os.listdir(cyclegan_dir)):

	NCE_path = os.path.join(NCE_dir, NCE_path)
	cyclegan_path = os.path.join(cyclegan_dir, cyclegan_path)

	NCE_wav, _ = read_wav(NCE_path, sr=sampling_rate)
	cyclegan_wav, _ = read_wav(cyclegan_path, sr=sampling_rate)

	NCE_list.append(NCE_wav)
	cyclegan_list.append(cyclegan_wav)

target_list = []
for target_path in os.listdir(target_dir):
	target_path = os.path.join(target_dir, target_path)
	target_wav, _ = read_wav(target_path, sr=sampling_rate)
	target_list.append(target_wav)

# NCE_list, cyclegan_list, target_list = [], [] ,[]
# for NCE_path, cyclegan_path, target_path in zip(os.listdir(NCE_dir), os.listdir(cyclegan_dir), os.listdir(target_dir)):
# 	NCE_path = os.path.join(NCE_dir, NCE_path)
# 	cyclegan_path = os.path.join(cyclegan_dir, cyclegan_path)
# 	target_path = os.path.join(target_dir, target_path)

# 	NCE_wav, _ = read_wav(NCE_path, sr=sampling_rate)
# 	cyclegan_wav, _ = read_wav(cyclegan_path, sr=sampling_rate)
# 	target_wav, _ = read_wav(target_path, sr=sampling_rate)

# 	NCE_list.append(NCE_wav)
# 	cyclegan_list.append(cyclegan_wav)
# 	target_list.append(target_wav)

# length = min(len(NCE_list), len(cyclegan_list), len(target_list))

spk_embeds_NCE = np.array([encoder.embed_speaker(NCE_list)])
spk_embeds_cyclegan = np.array([encoder.embed_speaker(cyclegan_list)])
spk_embeds_target = np.array([encoder.embed_speaker(target_list)])

spk_sim_NCE = np.inner(spk_embeds_NCE, spk_embeds_target)
spk_sim_cyclegan = np.inner(spk_embeds_cyclegan, spk_embeds_target)

print('NCE:{} CycleGAN:{}'.format(spk_sim_NCE, spk_sim_cyclegan))


