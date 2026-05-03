# ============================================================
# EEG PREPROCESSING PIPELINE
# Praktikum version
#
# Steps:
# 1. Define paths and condition
# 2. Load raw BrainVision EEG files
# 3. Set electrode montage
# 4. Visually inspect data and mark bad channels/segments
# 5. Interpolate bad channels
# 6. Remove 50 Hz line noise and bandpass filter
# 7. Concatenate blocks per subject
# 8. Run ICA for ocular artifact correction
# 9. Re-reference, resample, and save preprocessed data
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
from pathlib import Path
import copy

import mne
from meegkit import dss

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


# ============================================================
# PATHS
# ============================================================

default_path = Path.cwd()

eeg_raw_path = default_path / 'data' / 'raw'
save_path = default_path / 'data' / 'preprocessed'


# ============================================================
# GLOBAL PARAMETERS
# ============================================================

# Experimental conditions:
# a1 = azimuth, target right
# a2 = azimuth, target left
# e1 = elevation, target bottom
# e2 = elevation, target top
conditions = ['a1', 'a2', 'e1', 'e2']

# Select condition to preprocess
condition = conditions[0]


# ============================================================
# 1. LOAD RAW EEG FILES
# ============================================================

eeg_data_dict = {}

for subs in eeg_raw_path.iterdir():
    sub_name = subs.name
    sub_data_list = []

    for eeg_files in subs.iterdir():
        if eeg_files.suffix == '.vhdr' and condition in eeg_files.name:
            eeg_raw = mne.io.read_raw_brainvision(eeg_files, preload=True)
            sub_data_list.append(eeg_raw)

    eeg_data_dict[sub_name] = sub_data_list


# ============================================================
# 2. SET ELECTRODE MONTAGE
# ============================================================

for sub, eeg_list in eeg_data_dict.items():
    if eeg_list:
        for eeg_data in eeg_list:
            print(eeg_data)
            eeg_data.set_montage('standard_1020')


# ============================================================
# 3. VISUAL INSPECTION
#
# In the plot:
# - mark very noisy/dead channels as bad
# - mark strong artifact segments as BAD annotations
# - close the figure to continue
# ============================================================

for sub, eeg_list in eeg_data_dict.items():
    if eeg_list:
        for eeg_data in eeg_list:
            print(f"Inspecting {sub}")
            eeg_data.plot(block=True)


# ============================================================
# 4. INTERPOLATE BAD CHANNELS
#
# This uses the bad channels marked manually during inspection.
# A copy is created so the original raw object is not overwritten.
# ============================================================

eeg_interp_dict = {}

for sub, eeg_list in eeg_data_dict.items():
    eeg_interp_list = []

    if eeg_list:
        for eeg_data in eeg_list:
            eeg_interp = eeg_data.copy()
            eeg_interp.interpolate_bads()
            eeg_interp_list.append(eeg_interp)

    eeg_interp_dict[sub] = eeg_interp_list


# ============================================================
# 5. FILTERING
#
# First: remove 50 Hz line noise using DSS
# Then: bandpass filter from 1–30 Hz
# ============================================================

eeg_filt_dict = {}

for sub, eeg_interp_list in eeg_interp_dict.items():
    eeg_filt_list = []

    if eeg_interp_list:
        for eeg_data in eeg_interp_list:
            eeg_filter = eeg_data.copy()

            eeg_notch, iterations = dss.dss_line(
                eeg_data.get_data().T,
                fline=50,
                sfreq=eeg_data.info["sfreq"],
                nfft=400
            )

            eeg_filter._data = eeg_notch.T

            hi_filter = 1
            lo_filter = 30

            eeg_filtered = eeg_filter.copy().filter(
                l_freq=hi_filter,
                h_freq=lo_filter
            )

            eeg_filt_list.append(eeg_filtered)

    eeg_filt_dict[sub] = eeg_filt_list


# ============================================================
# 6. CONCATENATE BLOCKS PER SUBJECT
#
# All blocks belonging to the selected condition are concatenated
# before ICA.
# ============================================================

eeg_concat_dict = {}

for sub, eeg_filt_list in eeg_filt_dict.items():
    if eeg_filt_list:
        eeg_concat = mne.concatenate_raws(eeg_filt_list.copy())
        eeg_concat_dict[sub] = eeg_concat
    else:
        eeg_concat_dict[sub] = None


# ============================================================
# 7. ICA, RE-REFERENCE, RESAMPLE, SAVE
#
# For each subject:
# - fit ICA
# - inspect components and sources
# - manually select components to remove
# - apply ICA
# - add FCz if needed
# - average reference
# - resample to 125 Hz
# - save preprocessed file
# ============================================================

for subs in eeg_concat_dict.keys():

    if eeg_concat_dict[subs] is None:
        print(f"Skipping {subs}: no data found for condition {condition}")
        continue

    print(f"Running ICA for {subs}")

    eeg_ica = eeg_concat_dict[subs].copy()

    # Fit ICA
    ica = mne.preprocessing.ICA(method='fastica')
    ica.fit(eeg_ica)

    # Inspect ICA components
    ica.plot_components()
    ica.plot_sources(eeg_ica, block=True)

    # Apply ICA after manually selecting bad components
    ica.apply(eeg_ica)

    # Re-reference
    eeg_ica.add_reference_channels('FCz')  # check whether FCz was original reference
    eeg_ica.set_eeg_reference(ref_channels='average')

    # Resample
    eeg_ica.resample(sfreq=125)

    # Save
    os.makedirs(save_path / subs, exist_ok=True)

    eeg_ica.save(
        save_path / subs / f'{subs}_{condition}_concat_ica-raw.fif',
        overwrite=True)