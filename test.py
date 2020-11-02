"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import yaml
from options.test_options import TestOptions
from data import create_dataset, preprocess
from data.wav_folder import make_dataset, read_wav, write_wav, trun_spec, process_utterance
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util
import numpy as np
import librosa
import torch
from sklearn.preprocessing import StandardScaler
from parallel_wavegan.utils import load_model, read_hdf5
from tqdm import tqdm

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")



if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    sampling_rate = 24000

    model = create_model(opt)      # create a model given opt.model and other options
    netG = model.get_current_model()

    # load config
    if opt.config is None:
        dirname = os.path.dirname(opt.vocoder_dir)
        opt.config = os.path.join(dirname, "config.yml")
    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(opt))

    # restore scaler
    scaler = StandardScaler()
    if config["format"] == "hdf5":
        scaler.mean_ = read_hdf5(opt.stats, "mean")
        scaler.scale_ = read_hdf5(opt.stats, "scale")
    elif config["format"] == "npy":
        scaler.mean_ = np.load(opt.stats)[0]
        scaler.scale_ = np.load(opt.stats)[1]
    else:
        raise ValueError("support only hdf5 or npy format.")
    # from version 0.23.0, this information is needed
    scaler.n_features_in_ = scaler.mean_.shape[0]

    vocoder = load_model(opt.vocoder_dir, config)
    vocoder.remove_weight_norm()
    vocoder = vocoder.eval().to(device)

    print("Generating validation data B from A...")
    for i, file in enumerate(tqdm(sorted(os.listdir(opt.validation_A_dir)))):
        if i == 0:
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            model.parallelize()
            if opt.eval:
                model.eval()
        if i < 24:
	        filePath = os.path.join(opt.validation_A_dir, file)
	        wav, _ = read_wav(wav_path=filePath, sr=sampling_rate, mono=True)

	        # extract feature
	        wav, mel_spec = process_utterance(wav)
			
	        mel_spec = torch.from_numpy(mel_spec).unsqueeze(0).to(device).float()

	        with torch.no_grad():
	            converted_mel_spec = netG(mel_spec).squeeze(0)

			# normalize
	        converted_mel_spec = scaler.transform(converted_mel_spec.t().cpu().numpy())
	        converted_mel_spec = torch.from_numpy(converted_mel_spec).to(device).float()
	        with torch.no_grad():
	            wav_converted = vocoder.inference(converted_mel_spec).cpu().numpy()

	        if not os.path.exists(opt.output_A_dir):
	            os.makedirs(opt.output_A_dir)
	        write_wav(y=wav_converted, wav_path=os.path.join(opt.output_A_dir, os.path.basename(file)), sr=sampling_rate)

