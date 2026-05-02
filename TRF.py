import os
from pathlib import Path
import numpy as np
import mne
from mtrf import TRF
from mtrf.stats import crossval
import pandas as pd


default_path = Path.cwd()
data_path = default_path / 'data'
input_path = data_path / 'TRF' / 'input'
output_path = data_path / 'TRF' / 'output'

planes = ['azimuth', 'elevation']
sfreq = 125

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

from scipy.stats import zscore

for files in input_path.iterdir():
    sub_name = files.name[:5]

    if sub_name not in subs_matrices:
        subs_matrices[sub_name] = {}

    if conditions[0] in files.name:
        cond1_matrix = np.load(files, allow_pickle=True)
        # zscore across each column of predictors
        subs_matrices[sub_name]['cond1_predictors'] = cond1_matrix['predictors']

        # zscore across channels
        subs_matrices[sub_name]['cond1_eeg'] = cond1_matrix['eeg']

    elif conditions[2] in files.name:
        cond2_matrix = np.load(files, allow_pickle=True)
        # zscore across each column of predictors
        subs_matrices[sub_name]['cond2_predictors'] = cond2_matrix['predictors']
        # zscore across channels
        subs_matrices[sub_name]['cond2_eeg'] = cond2_matrix['eeg']


# concatenate cond1 + cond2 per subject and zscore
for sub_name, sub_data in subs_matrices.items():
    has_cond1 = all(k in sub_data for k in ['cond1_predictors', 'cond1_eeg'])
    has_cond2 = all(k in sub_data for k in ['cond2_predictors', 'cond2_eeg'])

    if has_cond1 and has_cond2:
        # --- concatenate ---
        predictors_concat = np.concatenate([sub_data['cond1_predictors'], sub_data['cond2_predictors']], axis=0)
        eeg_concat = np.concatenate([sub_data['cond1_eeg'], sub_data['cond2_eeg']], axis=0)
        print(sub_name)
        print("Predictors:", predictors_concat.shape)
        print("EEG:", eeg_concat.shape)

    elif has_cond1:
        print(f"{sub_name}: only cond1 available")
        predictors_concat = sub_data['cond1_predictors']
        eeg_concat = sub_data['cond1_eeg']

    elif has_cond2:
        print(f"{sub_name}: only cond2 available")
        predictors_concat = sub_data['cond2_predictors']
        eeg_concat = sub_data['cond2_eeg']

    else:
        print(f"Skipping {sub_name}: no valid data")
        continue

    predictors_concat = zscore(predictors_concat, axis=0)
    eeg_concat = zscore(eeg_concat, axis=0)


    # --- store ---
    subs_matrices[sub_name]['predictors'] = predictors_concat
    subs_matrices[sub_name]['eeg'] = eeg_concat
    subs_matrices[sub_name]['predictor_columns'] = predictor_columns

    print(sub_name)
    print("Predictors:", predictors_concat.shape)
    print("EEG:", eeg_concat.shape)
    X = predictors_concat
    Y = eeg_concat

    lambdas = np.logspace(-2, 2, 20)  # based on prev literature

    n_blocks = 5

    X_blocks = np.array_split(X, n_blocks, axis=0)
    Y_blocks = np.array_split(Y, n_blocks, axis=0)

    print("X blocks:", [x.shape for x in X_blocks])
    print("Y blocks:", [y.shape for y in Y_blocks])

    lambda_scores = {}

    for lmbda in lambdas:
        fwd_trf = TRF(direction=1)

        r = crossval(
            fwd_trf,
            X_blocks,
            Y_blocks,
            sfreq,
            tmin=-0.1,
            tmax=1.0,
            regularization=lmbda)

        lambda_scores[lmbda] = np.mean(r)
        print(f"lambda={lmbda}: mean r={np.mean(r):.4f}")

    best_lambda = max(lambda_scores, key=lambda_scores.get)

    print("Best lambda:", best_lambda)
    print("Best mean r:", lambda_scores[best_lambda])

    trf = TRF(direction=1, method='ridge')  # forward model

    trf.train(stimulus=X, response=Y, fs=sfreq, tmin=-0.1, tmax=1.0, regularization=best_lambda, seed=42)

    predictions, r = trf.predict(stimulus=X, response=Y, average=False)

    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use('TkAgg')

    plt.ion()

    # average across channels → (lags,)
    w0 = np.mean(trf.weights[0], axis=1)
    w1 = np.mean(trf.weights[1], axis=1)

    # --- smoothing ---
    window_len = 11
    hamming_win = np.hamming(window_len)
    hamming_win /= hamming_win.sum()

    w0_smooth = np.convolve(w0, hamming_win, mode='same')
    w1_smooth = np.convolve(w1, hamming_win, mode='same')

    # time axis
    lags = trf.times

    plt.figure()
    plt.plot(lags, w0_smooth, label='Target')
    plt.plot(lags, w1_smooth, label='Distractor')
    plt.axvline(0, linestyle='--')
    plt.xlim(-0.1, 0.5)
    plt.xlabel('Time lag (s)')
    plt.ylabel('TRF weight')
    plt.title('TRF comparison')
    plt.legend()
    plt.show()

    # RUN CLUSTER BASED NON PARAMETRIC PERMUTATION ACROSS CHANNELS








