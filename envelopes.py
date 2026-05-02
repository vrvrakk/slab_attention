from pathlib import Path
import numpy as np
import pandas as pd
import mne


def filter_block(block_data, condition):
    """Select block rows belonging to the current condition."""

    if condition == "a1":
        return block_data[
            (block_data["block_seq"] == "s1") &
            (block_data["block_condition"] == "azimuth")]

    elif condition == "a2":
        return block_data[
            (block_data["block_seq"] == "s2") &
            (block_data["block_condition"] == "azimuth")]

    elif condition == "e1":
        return block_data[
            (block_data["block_seq"] == "s1") &
            (block_data["block_condition"] == "elevation")]

    elif condition == "e2":
        return block_data[
            (block_data["block_seq"] == "s2") &
            (block_data["block_condition"] == "elevation")]

    else:
        raise ValueError(f"Unknown condition: {condition}")

# ============================================================
# LOAD DATA
# ============================================================


def load_stream_events(stream_name):
    """
    Load event arrays for one stream.

    Expected files contain names such as:
    stream1
    stream2
    """

    stream_events = []

    for file in sorted(events_path.iterdir()):
        if stream_name in file.name:
            event_array = np.load(file, allow_pickle=True)
            stream_events.append(event_array)

    return stream_events


def load_eeg_files(condition):
    """Load EEG files for one subject and condition."""

    eeg_files = []
    events_list = []
    events_id_list = []
    for file in sorted(eeg_path.iterdir()):
        if file.suffix == ".vhdr" and condition in file.name:
            raw = mne.io.read_raw_brainvision(file, preload=True)
            raw.set_montage('standard_1020')
            # raw.set_eeg_reference("average")
            raw.resample(sfreq=sfreq)
            raw_events, events_id = mne.events_from_annotations(raw)
            events_list.append(raw_events)
            events_id_list.append(events_id)
            eeg_files.append(raw)
    return eeg_files, events_list, events_id_list


def get_voices_dict():
    """
    Create dictionary matching each voice folder to stimulus numbers.

    Example:
    voices_dict['voice1']['one'] = 1
    """
    wav_nums = [1, 2, 3, 4, 5, 6, 8, 9]
    voices_dict = {}

    for voice_folder in envelopes_path.iterdir():

        if not voice_folder.is_dir():
            continue

        voice_dict = {}

        for number, wav_file in zip(wav_nums, sorted(voice_folder.iterdir())):
            voice_dict[wav_file.stem] = number
        voices_dict[voice_folder.stem] = voice_dict

    return voices_dict


# ============================================================
# BUILD ENVELOPE PREDICTORS
# ============================================================

def insert_envelope(predictor, number, onset, voice, eeg_len, voices_dict):
    """
    Insert one sound envelope into the predictor array.
    """
    for key, value in voices_dict[voice].items():
        if value == number:
            matching_keys = key
            print(f'Extracting {voice} envelope {key} for number {value}')
            if not matching_keys:
                print(f"No sound found for number {number} in {voice}")
                return

            sound_name = matching_keys
            envelope_file = envelopes_path / voice / f"{sound_name}.npy"

            envelope = np.load(envelope_file)

            end = min(onset + len(envelope), eeg_len)

            predictor[onset:end] = envelope[:end - onset]


def create_stream_predictors(eeg_files, voices_array, voices_dict, stream_label=''):
    """
    Create envelope predictors for one stream.

    Returns:
    - concatenated envelope array predictor
    """

    block_predictors = {}

    for i, (voice, stream_block) in enumerate(zip(voices_array, streams_dict.keys())):

        selected_stream = streams_dict[stream_block][stream_label]
        eeg_len = eeg_files[i].get_data().shape[1]
        predictor = np.zeros(eeg_len)

        for event in selected_stream:

            onset = int(event[0])
            number = int(event[2])
            if number in list(stream2_ids.keys()):
                number = stream2_ids[number]
            else:
                number = number


            insert_envelope(
                predictor=predictor,
                number=number,
                onset=onset,
                voice=voice,
                eeg_len=eeg_len,
                voices_dict=voices_dict)
        block_predictors[i] = predictor

    predictor_concat = np.concatenate(list(block_predictors.values()))

    save_predictor_concat(predictor_concat, stream_label=stream_label)

    return predictor_concat


# ============================================================
# SAVE CONCATENATED PREDICTORS
# ============================================================

def save_predictor_concat(predictor_concat, stream_label=''):
    """Save concatenated predictor."""

    save_path = predictors_path / "envelopes" / sub
    save_path.mkdir(parents=True, exist_ok=True)

    filename = f"{sub}_{condition}_{stream_label}_envelope_concat.npz"

    np.savez(save_path / filename, envelope=predictor_concat)

    print(f"Saved concatenated predictor: {filename}")


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":

    # ============================================================
    # USER SETTINGS
    # ============================================================

    PROJECT_ROOT = Path.cwd()

    sub = "sub30"
    condition = "a1"  # options: [a1, a2, e1, e2]

    sfreq = 125
    stim_dur = 0.745

    eeg_path = PROJECT_ROOT / "data" / "raw" / sub
    events_path = PROJECT_ROOT / "data" / "events" / sub
    blocks_path = PROJECT_ROOT / "data" / 'blocks'
    envelopes_path = PROJECT_ROOT / "voices_english" / "downsampled"
    predictors_path = PROJECT_ROOT / "data" / "predictors"
    predictors_path.mkdir(parents=True, exist_ok=True)

    print(f"Creating envelope predictors for {sub}, {condition}")

    # Load block information
    block_file = blocks_path / f"{sub}.csv"
    block_data = pd.read_csv(block_file)

    condition_block = filter_block(block_data, condition)
    voices_array = list(condition_block["Voices"])

    # stream2 map
    stream2_ids = {65:1, 66:2, 67:3, 68:4, 69:5, 70:6, 72:8, 73:9}

    # Load EEG
    eeg_files, events_list, events_id_list = load_eeg_files(condition)

    def define_streams(condition, stream1, stream2):
        if condition in ['a1', 'e1']:
            target = stream1
            distractor = stream2
        elif condition in ['a2', 'e2']:
            target = stream2
            distractor = stream1
        return target, distractor

    # filter events by stream:
    streams_dict = {}
    for block_index, event_list in enumerate(events_list):
        stream1 = []
        stream2 = []
        for event in event_list:
            if event[2] in np.arange(65, 74, 1):
                if event[2] == 71:
                    continue
                else:
                    stream2.append(event)
            elif event[2] in np.arange(1,10, 1):
                if event[2] == 7:
                    continue
                else:
                    stream1.append(event)
            else:
                continue
        target, distractor = define_streams(condition, stream1, stream2)
        streams_dict[block_index] = {'target': stream1, 'distractor': stream2}

    voices_dict = get_voices_dict()

    # Create stream predictors
    target_stream_concat = create_stream_predictors(
        eeg_files=eeg_files,
        voices_array=voices_array,
        voices_dict=voices_dict,
        stream_label='target')

    distractor_stream_concat = create_stream_predictors(
        eeg_files=eeg_files,
        voices_array=voices_array,
        voices_dict=voices_dict,
        stream_label='distractor')
