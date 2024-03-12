import torchaudio
import os
import torch
import random
from torch.nn import functional as F
from torchaudio.transforms import SpeedPerturbation
import speechaugs

def weak_augment_with_time_mask(input_values):
    num_masks = 2
    input_values = torch.Tensor(input_values).unsqueeze(0)
    mask_lengths = [int(input_values.shape[1] * 0.1) for _ in range(num_masks)]
    mask_positions = [torch.randint(0, input_values.shape[1] - mask_lengths[i], (1,)) for i in range(num_masks)]
    mask = torch.ones_like(input_values)
    # mask out random positions
    for i in range(num_masks):
        mask[:, mask_positions[i]:mask_positions[i] + mask_lengths[i]] = 0
    weak_augmented_input_values = input_values * mask
    return weak_augmented_input_values.tolist()[0]

def strong_augment_with_speed_perturbation(input_values):
    input_values = torch.Tensor(input_values).unsqueeze(0)
    speed_perturb = SpeedPerturbation(16000, [0.8, 0.9, 1.1, 1.2])
    input_values_pertubed = speed_perturb(input_values)
    return input_values_pertubed[0].tolist()[0]

def strong_augment_with_pitch_shift(input_values):
    input_values = torch.Tensor(input_values).unsqueeze(0)
    lower_pitch = [-5, -3]
    higher_pitch = [3, 5]
    # either lower or higher pitch
    if random.random() < 0.5:
        input_values_pertubed = speechaugs.PitchShiftLibrosa(p=1., sr=16000, min_steps=min(lower_pitch), max_steps=max(lower_pitch))(waveform=input_values)['waveform']
    else:
        input_values_pertubed = speechaugs.PitchShiftLibrosa(p=1., sr=16000, min_steps=min(higher_pitch), max_steps=max(higher_pitch))(waveform=input_values)['waveform']
    return input_values_pertubed.tolist()[0]

def strong_augment_with_noise(input_values, ratio=1.0):
    input_values = torch.Tensor(input_values).unsqueeze(0)
    noise_files_dir = 'path to the ESC 50 dataset\'s audio dir'
    noise_file = random.choice(os.listdir(noise_files_dir))
    noise_file_path = os.path.join(noise_files_dir, noise_file)
    noise_audio, _ = torchaudio.load(noise_file_path)
    # if noise file is shorter than input_values, interpolate it
    if noise_audio.shape[1] < input_values.shape[1]:
        noise_audio = F.interpolate(noise_audio.unsqueeze(1), size=input_values.shape[1], mode='linear', align_corners=False).squeeze(1)
    # take random position from noise file
    random_noise_start = random.randint(0, noise_audio.shape[1] - input_values.shape[1])
    noise_to_add = noise_audio[:, random_noise_start:random_noise_start + input_values.shape[1]]
    assert noise_to_add.shape == input_values.shape
    # add noise to input_values with ratio
    strong_augmented_input_values = input_values + noise_to_add * ratio
    return strong_augmented_input_values.tolist()[0]
