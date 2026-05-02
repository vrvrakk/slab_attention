import os
from pathlib import Path
import numpy as np
import mne

PROJECT_ROOT = Path.cwd()
data_path = PROJECT_ROOT / 'data'
bad_segments_path = data_path / 'bad_segments'
envelopes_arr_path = data_path / "predictors"
button_press_path = data_path / 'button_press'
eeg_path = data_path / 'preprocessed'

# save path:
save_path = data_path / 'TRF' / 'input'
save_path.mkdir(parents=True, exist_ok=True)

# condition:
condition = 'a1'
sub = 'sub30'
env_roi = np.array(['Cz', 'FCz', 'CPz', 'Fz'])  # main env roi

# load EEG files:
eeg_dict = {}
for sub_fold in eeg_path.iterdir():
    for sub_files in sub_fold.iterdir():
        if condition in sub_files.name:
            eeg = mne.io.read_raw_fif(sub_files, preload=True)
            eeg.pick_channels(env_roi)

    eeg_dict[sub_fold.name] = eeg.get_data()  # data already downsampled

# load bad EEG segments array:
for files in bad_segments_path.iterdir():
    if condition in files.name:
        bads_dict = np.load(files, allow_pickle=True)
        bads_dict = bads_dict['data'].item()

# load button-press array:
for files in button_press_path.iterdir():
    if condition in files.name:
        buttons_dict = np.load(files, allow_pickle=True)
        buttons_dict = buttons_dict['data'].item()


# load envelopes array:
envs_dict = {}
for files in envelopes_arr_path.iterdir():
    for sub_name in files.iterdir():
        for sub_cond in sub_name.iterdir():
            if condition in sub_cond.name:
                if 'target' in sub_cond.name:
                    target_envs = np.load(sub_cond, allow_pickle=True)
                    target_envs = target_envs['envelope']
                elif 'distractor' in sub_cond.name:
                    distractor_envs = np.load(sub_cond, allow_pickle=True)
                    distractor_envs = distractor_envs['envelope']
    envs_dict[sub_name.name] = {'target': target_envs, 'distractor': distractor_envs}

# ensure lengths are correct:
print(len({eeg_dict[sub].shape[1], bads_dict[sub].size, buttons_dict[sub].size, envs_dict[sub]['target'].size, envs_dict[sub]['distractor'].size}) == 1)


# mask bad segments in predictors and EEG data of one subject, one condition:
mask = bads_dict[sub] == 999
buttons_masked = buttons_dict[sub][~mask]
target_envs_masked = envs_dict[sub]['target'][~mask]
distractor_envs_masked = envs_dict[sub]['distractor'][~mask]
eeg_masked = eeg_dict[sub][:, ~mask]

# create TRF input matrix for one subject
X = np.column_stack([
    target_envs_masked,
    distractor_envs_masked,
    buttons_masked])

Y = eeg_masked.T  # channels x samples -> samples x channels

print("X shape:", X.shape)
print("Y shape:", Y.shape)

assert X.shape[0] == Y.shape[0]


np.savez(save_path/f'{sub}_{condition}_matrix.npz',
         predictors=X,
         eeg=Y)

