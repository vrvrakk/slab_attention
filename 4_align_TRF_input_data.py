import os
from pathlib import Path
import numpy as np
import mne


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path.cwd()

data_path = PROJECT_ROOT / 'data'
bad_segments_path = data_path / 'bad_segments'
envelopes_arr_path = data_path / "predictors"
button_press_path = data_path / 'button_press'
eeg_path = data_path / 'preprocessed'

# save path:
save_path = data_path / 'TRF' / 'input'
save_path.mkdir(parents=True, exist_ok=True)


# ============================================================
# USER SETTINGS
# ============================================================

# condition:
condition = 'a1'
sub = 'sub30'

# main env roi; only these EEG channels will be used for the envelope TRF model
env_roi = np.array(['Cz', 'FCz', 'CPz', 'Fz'])


# ============================================================
# LOAD EEG DATA
# ============================================================

# load EEG files:
eeg_dict = {}

for sub_fold in eeg_path.iterdir():

    if not sub_fold.is_dir():
        continue

    eeg = None

    for sub_files in sub_fold.iterdir():

        if condition in sub_files.name:

            eeg = mne.io.read_raw_fif(sub_files, preload=True)

            # select only ROI channels
            eeg.pick_channels(env_roi)

    # data already downsampled
    if eeg is not None:
        eeg_dict[sub_fold.name] = eeg.get_data()


# ============================================================
# LOAD BAD-SEGMENT ARRAY
# ============================================================

# load bad EEG segments array:
# 0 = good segment
# -999 = bad segment, should be excluded from TRF input
bads_dict = None

for files in bad_segments_path.iterdir():
    if condition in files.name:
        bads_dict = np.load(files, allow_pickle=True)
        bads_dict = bads_dict['data'].item()


# ============================================================
# LOAD BUTTON-PRESS ARRAY
# ============================================================

# load button-press array:
# 0 = no response
# 1 = button press impulse
buttons_dict = None

for files in button_press_path.iterdir():
    if condition in files.name:
        buttons_dict = np.load(files, allow_pickle=True)
        buttons_dict = buttons_dict['data'].item()


# ============================================================
# LOAD ENVELOPE ARRAYS
# ============================================================

# load envelopes array:
# target envelope = attended stream envelope
# distractor envelope = unattended stream envelope
envs_dict = {}

for files in envelopes_arr_path.iterdir():

    if not files.is_dir():
        continue

    for sub_name in files.iterdir():

        if not sub_name.is_dir():
            continue

        target_envs = None
        distractor_envs = None

        for sub_cond in sub_name.iterdir():

            if condition in sub_cond.name:

                if 'target' in sub_cond.name:
                    target_envs = np.load(sub_cond, allow_pickle=True)
                    target_envs = target_envs['envelope']

                elif 'distractor' in sub_cond.name:
                    distractor_envs = np.load(sub_cond, allow_pickle=True)
                    distractor_envs = distractor_envs['envelope']

        if target_envs is not None and distractor_envs is not None:
            envs_dict[sub_name.name] = {
                'target': target_envs,
                'distractor': distractor_envs
            }


# ============================================================
# CHECK THAT ALL ARRAYS HAVE SAME LENGTH
# ============================================================

# ensure lengths are correct:
# EEG is channels x samples, so use eeg_dict[sub].shape[1]
same_length = len({
    eeg_dict[sub].shape[1],
    bads_dict[sub].size,
    buttons_dict[sub].size,
    envs_dict[sub]['target'].size,
    envs_dict[sub]['distractor'].size
}) == 1

print("All arrays same length:", same_length)

assert same_length, "EEG, bads, button press, target envelope, and distractor envelope do not have same length."


# ============================================================
# REMOVE BAD SEGMENTS
# ============================================================

# mask bad segments in predictors and EEG data of one subject, one condition:
# True = bad sample, False = good sample
mask = bads_dict[sub] == -999

# keep only good samples in all predictors
buttons_masked = buttons_dict[sub][~mask]
target_envs_masked = envs_dict[sub]['target'][~mask]
distractor_envs_masked = envs_dict[sub]['distractor'][~mask]

# keep only good samples in EEG
# EEG shape before masking: channels x samples
eeg_masked = eeg_dict[sub][:, ~mask]


# ============================================================
# CREATE TRF INPUT MATRICES
# ============================================================

# create TRF input matrix for one subject
# X = predictors, shape: samples x predictors
# predictor columns:
# 1. target envelope
# 2. distractor envelope
# 3. button press impulse
X = np.column_stack([
    target_envs_masked,
    distractor_envs_masked,
    buttons_masked
])

# Y = EEG response, shape: samples x channels
# original EEG is channels x samples, so transpose it
Y = eeg_masked.T

print("X shape:", X.shape)
print("Y shape:", Y.shape)

assert X.shape[0] == Y.shape[0], "X and Y do not have same number of samples."


# ============================================================
# SAVE TRF INPUT MATRIX
# ============================================================

np.savez(
    save_path / f'{sub}_{condition}_matrix.npz',
    predictors=X,
    eeg=Y)