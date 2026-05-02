# SAVING AND PATH CREATION
import os
from pathlib import Path
# TO SAVE PRE-PROCESSED EEG DATA TEMPORARILY AS A NEW VARIABLE, IN ORDER TO NOT MUTATE ORIGINAL EEG DATA VARIABLE
import copy
import mne # your main EEG pre-processing package/library
from meegkit import dss # for filtering EEG data
# YOU BASICALLY ALMOST ALWAYS IMPORT THESE
import numpy as np # to handle data, arrays, do some transformations, etc...
import pandas as pd # to handle dataframes
# PLOTTING
import matplotlib # essential for creating plots and figures
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

# define paths for loading files and saving files:
# first the default path:
default_path = Path.cwd()
eeg_raw_path = default_path / 'data' / 'raw'
save_path = default_path / 'data' / 'preprocessed'

# define some global parameters / fixed info that will be used along the pre-processing pipelines/steps

# block conditions defined: a1 target was right, a2 left & e1 bottom, e2 top
conditions = ['a1', 'a2', 'e1', 'e2']
condition = conditions[0]  # select one condition to work with (0-3)

# load raw eeg files, and separate by subject, and within-subject by plane.
eeg_data_dict = {}
for subs in eeg_raw_path.iterdir():
    sub_name = subs.name
    sub_data_list = []
    for eeg_files in subs.iterdir():
        if 'vhdr' in eeg_files.name:
            if condition in eeg_files.name:
                eeg_raw = mne.io.read_raw_brainvision(eeg_files, preload=True)
                sub_data_list.append(eeg_raw)
    eeg_data_dict[sub_name] = sub_data_list

# add electrodes montage:
for sub, eeg_list in eeg_data_dict.items():
        if eeg_list:  # handle if not empty list
            for eeg_data in eeg_list:
                print(eeg_data)
                eeg_data.set_montage('standard_1020')

# interpolate bad channels (if any) by:
# 1. plot eeg
# 2. visually select channels that look like hell - if applicable
# 3. close figure
# 4. interpolate
for sub, eeg_list in eeg_data_dict.items():
        if eeg_list:
            for eeg_data in eeg_list:
                eeg_data.plot(block=True)
                # here you can visually investigate, select BAD channels (extreme noise OR dead), or select
                # bad segments (click A on keyboard and then ADD SEGMENT. BAD
                # to activate bad segment selecting, and select desired area)
                # which distort the signal with obvious noise across ALL channels (like jaw clenching,
                # big bodily movements, etc); transient noise (slow drifts will be removed with bandpass filtering)

eeg_interp_dict = {}
for sub, eeg_list in eeg_data_dict.items():
    eeg_interp_list = []
    if eeg_list:
        for eeg_data in eeg_list:
            eeg_interp = eeg_data.copy()  # copy eeg data to prevent overwrite of raw
            eeg_interp.interpolate_bads() # interpolates channels that were selected (if any)
            eeg_interp_list.append(eeg_interp)
        # store list under the correct subject name
    eeg_interp_dict[sub] = eeg_interp_list

# 8. 50Hz notch & bandpass filter with range of choice: 1-30 Hz
eeg_filt_dict = {}
for sub, eeg_interp_llist in eeg_interp_dict.items():
    eeg_filt_list = []
    if eeg_interp_list:
        for eeg_data in eeg_interp_list:
            eeg_filter = eeg_data.copy()
            data = mne.io.RawArray(data=eeg_data.get_data(), info=eeg_data.info)
            eeg_notch, iterations = dss.dss_line(eeg_data.get_data().T, fline=50,
                                                 sfreq=data.info["sfreq"],
                                                 nfft=400)

            eeg_filter._data = eeg_notch.T
            hi_filter = 1
            lo_filter = 30
            eeg_filtered = eeg_filter.copy().filter(hi_filter, lo_filter)
            eeg_filt_list.append(eeg_filtered)
    eeg_filt_dict[sub] = eeg_filt_list

# concatenate subject data, for each condition/plane respectively:

#  apply ICA: we use this to remove any ocular artifacts -> blinks, eye movements. if you are unsure about a component
#  do NOT remove it
eeg_concat_dict = {}

for sub, eeg_filt_list in eeg_filt_dict.items():
    if eeg_filt_list:
    # concatenate only if we found matching files
        eeg_concat = mne.concatenate_raws(eeg_filt_list.copy())
        eeg_concat_dict[sub] = eeg_concat
    else:
        eeg_concat_dict[sub] = None

##################
# repeat ICA application for each sub
for subs in eeg_concat_dict.keys():
    eeg_ica = eeg_concat_dict[subs].copy()
    # a. fit ICA:
    ica = mne.preprocessing.ICA(method='fastica')
    ica.fit(eeg_ica)  # bad segments that were marked in the EEG signal will be excluded.
    # b. investigate...:
    ica.plot_components()
    ica.plot_sources(eeg_ica, block=True)
    # c. apply ICA to remove selected components: blinks, eye movements etc.
    ica.apply(eeg_ica)
    # d. re-reference with average:
    eeg_ica.add_reference_channels('FCz') # pls ensure this is correct with new electrode set
    eeg_ica.set_eeg_reference(ref_channels='average')
    eeg_ica.resample(sfreq=125)
    # save pre-processed EEG data:
    os.makedirs(save_path / sub, exist_ok=True)
    eeg_ica.save(save_path / sub / f'{sub}_{condition}_concat_ica-raw.fif', overwrite=True)



