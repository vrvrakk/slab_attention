'''
With this script EEG events are extracted according to the stream of interest; then they are saved, as well as separate
event arrays for only target and distractor streams respectively;
then two arrays are created, aligned with the EEG data length:

1- bad segment array - specifies which area of the EEG data is marked as bad, and adds '999' as the value; rest is 0,
indicating -> good segment -> bad segments will NOT be used when running the main TRF analysis

2- responses array - specifies when a button-press response took place - based on stream event markers; if response at
samplepoint x took place, add 1; otherwise 0 -> this is a binary impulse array and will be used for the TRF analysis ->
as a way to track when a motor response took place

'''

from pathlib import Path
import os
import pandas as pd
import numpy as np
import mne

default_path = Path.cwd()
events_map_path = default_path / 'misc'
preprocessed_eeg = default_path / 'data' / 'preprocessed'
save_path = default_path / 'data'

# a dictionary containing the stimuli events markers as 'keys' and the actual values/names we want for them:
import ast
with open(events_map_path/'stim_event_markers.txt', 'r') as f:
    events_mapping = ast.literal_eval(f.read())
with open(events_map_path/'corresponding_nums.txt', 'r') as f:
    corresponding_nums = ast.literal_eval(f.read())

# block conditions defined: a1 target was right, a2 left & e1 bottom, e2 top
conditions = ['a1', 'a2', 'e1', 'e2']
condition = conditions[0]  # select one condition to work with - 0-3
sfreq = 125  # sample rate of the EEG data

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

# load pre-processed EEG data:
eeg_dict = {}
events_dict = {}
target_events_dict = {}
distractor_events_dict = {}
response_events_dict = {}
for sub_fold in preprocessed_eeg.iterdir():
    for sub_files in sub_fold.iterdir():
        if condition in sub_files.name:
            eeg = mne.io.read_raw_fif(sub_files, preload=True)
            eeg_dict[sub_fold.name] = eeg
            # extract events, and event IDs:
            events, event_ids = mne.events_from_annotations(eeg)
            for idx, event in enumerate(events):
                # find event key in event_ids:
                event_key = [key for key, val in event_ids.items() if event[2] == val][0]
                # get correct event key from events_mapping:
                if event_key in list(events_mapping.keys()):
                    correct_event_val = events_mapping[event_key]
                    events[idx, 2] = correct_event_val
                else:
                    correct_event_val = 0
                    events[idx, 2] = correct_event_val
            # replace current event_ids with correct event mapping dictionary:
            for event_key, event_value in event_ids.items():
                if event_key in list(events_mapping.keys()):
                    correct_event_val = events_mapping[event_key]
                    event_ids[event_key] = correct_event_val

            event_ids = {key: val for key, val in event_ids.items() if key in list(events_mapping.keys())}
            events = [event for event in events if event[2] in (event_ids.values())]
            # separate events also based on target & distractor stream, as well as button press responses
            target_stream_events = [event for event in events if event[2] in target_stream]
            target_events_dict[sub_fold.name] = target_stream_events

            distractor_stream_events = [event for event in events if event[2] in distractor_stream]
            for event in distractor_stream_events:
                if event[2] in corresponding_nums:
                    event[2] = corresponding_nums[event[2]]
            distractor_events_dict[sub_fold.name] = distractor_stream_events
            response_events = [event for event in events if event[2] in responses]
            for event in response_events:
                if event[2] in corresponding_nums:
                    event[2] = corresponding_nums[event[2]]
            response_events_dict[sub_fold.name] = response_events

            events_dict[sub_fold.name] = {'events': events, 'event_ids': event_ids}
            events_path = save_path / 'events' / sub_fold.name
            events_path.mkdir(parents=True, exist_ok=True)
            # save as zip
            # 1. all events
            np.savez(events_path/f'{condition}_events.npz', data=events_dict)
            # 2. target stream events
            np.savez(events_path/f'{condition}_target_events.npz', data=target_events_dict)
            # 3. distractor stream events
            np.savez(events_path/f'{condition}_distractor_events.npz', data=distractor_events_dict)
            # 4. response events
            np.savez(events_path/f'{condition}_response_events.npz', data=response_events_dict)

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
            offset_samples = int(np.round((offset) * sfreq))
            bad_series[onset_samples:offset_samples] = -999
    bad_series_dict[subs] = bad_series

bads_arr_save_path = save_path / 'bad_segments'
bads_arr_save_path.mkdir(parents=True, exist_ok=True)
np.savez(bads_arr_save_path/f'{condition}_bads.npz', data=bad_series_dict)

# also one for responses -> binary input -> 0s when no response, 1s when response took place
responses_arr_dicts = {}
for subs, eeg in eeg_dict.items():
    eeg_len = eeg.get_data().shape[1]
    response_arr = np.zeros(eeg_len)
    response_events = response_events_dict[subs]
    response_events = [response_event[0] for response_event in response_events]
    for idx, val in enumerate(response_arr):
        if idx in response_events:
            response_arr[idx] = 1
    responses_arr_dicts[subs] = response_arr

response_arr_save_path = save_path / 'button_press'
response_arr_save_path.mkdir(parents=True, exist_ok=True)
np.savez(response_arr_save_path/f'{condition}_response_predictor', data=responses_arr_dicts)







