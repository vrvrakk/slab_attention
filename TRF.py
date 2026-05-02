import os
from pathlib import Path
import numpy as np
import mne
from mtrf import TRF


default_path = Path.cwd()
data_path = default_path / 'data'
input_path = data_path / 'TRF' / 'input'
output_path = data_path / 'TRF' / 'output'

planes = ['azimuth', 'elevation']


def get_conditions(n=None):
    plane = planes[n]
    if plane == 'azimuth':
        conditions = ['a1', 'a2']
    elif plane == 'elevation':
        conditions = ['e1', 'e2']
    return conditions, plane


conditions, plane = get_conditions(0)

# load matrices of all subs, both conditions:
predictor_columns = ['target_env', 'distractor_env', 'button_presses']

subs_matrices = {}

for files in input_path.iterdir():
    sub_name = files.name[:5]

    if sub_name not in subs_matrices:
        subs_matrices[sub_name] = {}

    if conditions[0] in files.name:
        cond1_matrix = np.load(files, allow_pickle=True)
        subs_matrices[sub_name]['cond1_predictors'] = cond1_matrix['predictors']
        subs_matrices[sub_name]['cond1_eeg'] = cond1_matrix['eeg']

    elif conditions[2] in files.name:
        cond2_matrix = np.load(files, allow_pickle=True)
        subs_matrices[sub_name]['cond2_predictors'] = cond2_matrix['predictors']
        subs_matrices[sub_name]['cond2_eeg'] = cond2_matrix['eeg']


# concatenate cond1 + cond2 per subject
for sub_name, sub_data in subs_matrices.items():
    if all(k in sub_data for k in ['cond1_predictors', 'cond2_predictors', 'cond1_eeg', 'cond2_eeg']):
        predictors_concat = np.concatenate(
            [sub_data['cond1_predictors'], sub_data['cond2_predictors']], axis=0)

        eeg_concat = np.concatenate([sub_data['cond1_eeg'], sub_data['cond2_eeg']], axis=0)

        subs_matrices[sub_name]['predictors'] = predictors_concat
        subs_matrices[sub_name]['eeg'] = eeg_concat
        subs_matrices[sub_name]['predictor_columns'] = predictor_columns

        print(sub_name)
        print("Predictors:", predictors_concat.shape)
        print("EEG:", eeg_concat.shape)
    else:
        print(f"Skipping {sub_name}: missing one condition")







