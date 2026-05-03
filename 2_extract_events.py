'''

1- bad segment array - specifies which area of the EEG data is marked as bad, and adds '999' as the value; rest is 0,
indicating -> good segment -> bad segments will NOT be used when running the main TRF analysis

2- responses array - specifies when a button-press response took place - based on stream event markers; if response at
samplepoint x took place, add 1; otherwise 0 -> this is a binary impulse array and will be used for the TRF analysis ->
as a way to track when a motor response took place
'''

from pathlib import Path
import os
import numpy as np
import mne
import ast

# ============================================================
# PATHS
# ============================================================

default_path = Path.cwd()
events_map_path = default_path / 'misc'
preprocessed_eeg = default_path / 'data' / 'preprocessed'
save_path = default_path / 'data'


# ============================================================
# LOAD EVENT MAPPINGS
# ============================================================

# a dictionary containing the stimuli events markers as 'keys' and the actual values/names we want for them:
with open(events_map_path/'stim_event_markers.txt', 'r') as f:
    events_mapping = ast.literal_eval(f.read())

with open(events_map_path/'corresponding_nums.txt', 'r') as f:
    corresponding_nums = ast.literal_eval(f.read())


# ============================================================
# GLOBAL PARAMETERS
# ============================================================

# block conditions defined: a1 target was right, a2 left & e1 bottom, e2 top
conditions = ['a1', 'a2', 'e1', 'e2']
condition = conditions[0]  # select one condition to work with - 0-3
sfreq = 125  # sample rate of the EEG data


# ============================================================
# DEFINE STREAMS AND RESPONSES
# ============================================================

# depending on condition, select target / distractor stream:
# when condition contains 1, if azimuth (a1) or elevation (e1), the corresponding target stream event numbers are 1-9
# when condition contains 2, if azimuth (a2) or elevation (e2), the corresponding target stream event numbers are 65-73
stream1 = list(np.arange(1, 10, 1))
stream2 = list(np.arange(65, 74, 1))
responses = list(np.arange(129, 138, 1))  # button press event numbers

# whether it is 1-9, 65-73 or 129-137 -> they all represent numbers 1 to 9; different number order is used per stream,
# in order to be able to identify WHICH stream we are referring to

if condition in ['a1', 'e1']:
    target_stream = stream1
    distractor_stream = stream2
else:
    target_stream = stream2
    distractor_stream = stream1


# ============================================================
# LOAD EEG AND EXTRACT EVENTS
# ============================================================

# load pre-processed EEG data:
eeg_dict = {}
response_events_dict = {}

for sub_fold in preprocessed_eeg.iterdir():

    if not sub_fold.is_dir():
        continue

    for sub_files in sub_fold.iterdir():

        if condition in sub_files.name:

            eeg = mne.io.read_raw_fif(sub_files, preload=True)
            eeg_dict[sub_fold.name] = eeg  # save each subject's EEG data into a dictionary

            # extract events, and event IDs:
            events, event_ids = mne.events_from_annotations(eeg)

            # remap events to correct numerical coding
            for idx, event in enumerate(events):

                # find event key in event_ids:
                event_key = [key for key, val in event_ids.items() if event[2] == val]

                if not event_key:
                    continue

                event_key = event_key[0]

                # get correct event key from events_mapping:
                if event_key in events_mapping:
                    correct_event_val = events_mapping[event_key]
                else:
                    correct_event_val = 0

                events[idx, 2] = correct_event_val

            # replace current event_ids with correct event mapping dictionary:
            for event_key in list(event_ids.keys()):
                if event_key in events_mapping:
                    event_ids[event_key] = events_mapping[event_key]
                else:
                    del event_ids[event_key]

            # keep only relevant events
            events = np.array([event for event in events if event[2] in event_ids.values()])

            # extract response events
            response_events = [event for event in events if event[2] in responses]

            # remap response values to 1-9
            for event in response_events:
                if event[2] in corresponding_nums:
                    event[2] = corresponding_nums[event[2]]

            response_events_dict[sub_fold.name] = np.array(response_events)


# ============================================================
# CREATE BAD SEGMENT ARRAY
# ============================================================

# ADDITIONALLY -> create numpy array, where bad segments are marked == 999
bad_series_dict = {}

for subs, eeg in eeg_dict.items():

    eeg_len = eeg.get_data().shape[1]
    bad_series = np.zeros(eeg_len)

    for description, onset, duration in zip(eeg.annotations.description,
                                            eeg.annotations.onset,
                                            eeg.annotations.duration):

        if description.lower().startswith('bad'):

            offset = onset + duration

            onset_samples = int(np.round(onset * sfreq))
            offset_samples = int(np.round(offset * sfreq))

            # safety bounds
            onset_samples = max(onset_samples, 0)
            offset_samples = min(offset_samples, eeg_len)

            bad_series[onset_samples:offset_samples] = -999

    bad_series_dict[subs] = bad_series


# ============================================================
# SAVE BAD SEGMENTS
# ============================================================

bads_arr_save_path = save_path / 'bad_segments'
bads_arr_save_path.mkdir(parents=True, exist_ok=True)

# dictionary with the segments marked as 'bad' as an array is saved as a zip file; contains all subs
np.savez(bads_arr_save_path/f'{condition}_bads.npz', data=bad_series_dict)


# ============================================================
# CREATE RESPONSE ARRAY
# ============================================================

# also one for responses -> binary input -> 0s when no response, 1s when response took place
responses_arr_dicts = {}

for subs, eeg in eeg_dict.items():

    eeg_len = eeg.get_data().shape[1]
    response_arr = np.zeros(eeg_len)

    response_events = response_events_dict.get(subs, np.empty((0, 3)))

    if len(response_events) > 0:

        # extract sample indices directly (FASTER than looping)
        response_samples = response_events[:, 0].astype(int)

        # safety bounds
        response_samples = response_samples[
            (response_samples >= 0) & (response_samples < eeg_len)
        ]

        response_arr[response_samples] = 1

    responses_arr_dicts[subs] = response_arr


# ============================================================
# SAVE RESPONSE ARRAYS
# ============================================================

response_arr_save_path = save_path / 'button_press'
response_arr_save_path.mkdir(parents=True, exist_ok=True)

# dictionary of button press impulses array saved as a zip file; contains multiple subjects that can be accessed
np.savez(response_arr_save_path/f'{condition}_response_predictor.npz', data=responses_arr_dicts)