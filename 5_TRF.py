# ============================================================
# TRF ANALYSIS SCRIPT
#
# This script:
# 1. Loads TRF input matrices for all subjects
# 2. Combines the two sub-conditions belonging to one plane
#    e.g. azimuth = a1 + a2, elevation = e1 + e2
# 3. Z-scores predictors and EEG per subject
# 4. Optimizes ridge regularization per subject
# 5. Computes one group-level lambda
# 6. Refits each subject using the same group lambda
# 7. Saves target and distractor TRF weights
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
from pathlib import Path

import numpy as np
import pandas as pd
import mne

from scipy.stats import zscore

from mtrf import TRF
from mtrf.stats import neg_mse, pearsonr

# plotting
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()


# ============================================================
# PATHS
# ============================================================

# define your paths where you will load/save files
default_path = Path.cwd()

data_path = default_path / 'data'
input_path = data_path / 'TRF' / 'input'
output_path = data_path / 'TRF' / 'output'
output_path.mkdir(parents=True, exist_ok=True)


# ============================================================
# GLOBAL PARAMETERS
# ============================================================

planes = ['azimuth', 'elevation']
sfreq = 125

predictor_columns = [
    'target_env',
    'distractor_env',
    'button_presses'
]

lambdas = np.logspace(-2, 2, 20)  # regularization values based on previous literature
tmin, tmax = -0.1, 1.0

n_blocks = 5  # number of chunks used for regularization optimization


# ============================================================
# SELECT PLANE / CONDITIONS
# ============================================================

def get_conditions(n=None):
    """Return the two conditions belonging to one spatial plane."""

    plane = planes[n]

    if plane == 'azimuth':
        conditions = ['a1', 'a2']

    elif plane == 'elevation':
        conditions = ['e1', 'e2']

    else:
        raise ValueError(f"Unknown plane: {plane}")

    return conditions, plane


conditions, plane = get_conditions(0)


# ============================================================
# LOAD INPUT MATRICES
# ============================================================

# load matrices of all subs, both conditions:
subs_matrices = {}

for files in input_path.iterdir():

    # assumes filenames start with subject ID, e.g. sub30_a1_matrix.npz
    sub_name = files.name[:5]

    if sub_name not in subs_matrices:
        subs_matrices[sub_name] = {}

    if conditions[0] in files.name:

        cond1_matrix = np.load(files, allow_pickle=True)

        # predictors shape: samples x predictors
        subs_matrices[sub_name]['cond1_predictors'] = cond1_matrix['predictors']

        # EEG shape: samples x channels
        subs_matrices[sub_name]['cond1_eeg'] = cond1_matrix['eeg']

    elif conditions[1] in files.name:

        cond2_matrix = np.load(files, allow_pickle=True)

        # predictors shape: samples x predictors
        subs_matrices[sub_name]['cond2_predictors'] = cond2_matrix['predictors']

        # EEG shape: samples x channels
        subs_matrices[sub_name]['cond2_eeg'] = cond2_matrix['eeg']


# ============================================================
# CONCATENATE CONDITIONS AND Z-SCORE
# ============================================================

# concatenate cond1 + cond2 per subject and zscore
all_best_lambdas = {}

for sub_name, sub_data in subs_matrices.items():

    has_cond1 = all(k in sub_data for k in ['cond1_predictors', 'cond1_eeg'])
    has_cond2 = all(k in sub_data for k in ['cond2_predictors', 'cond2_eeg'])

    if has_cond1 and has_cond2:

        # concatenate the two sub-conditions belonging to the selected plane
        predictors_concat = np.concatenate(
            [sub_data['cond1_predictors'], sub_data['cond2_predictors']],
            axis=0
        )

        eeg_concat = np.concatenate(
            [sub_data['cond1_eeg'], sub_data['cond2_eeg']],
            axis=0
        )

    elif has_cond1:

        print(f"{sub_name}: only {conditions[0]} available")

        predictors_concat = sub_data['cond1_predictors']
        eeg_concat = sub_data['cond1_eeg']

    elif has_cond2:

        print(f"{sub_name}: only {conditions[1]} available")

        predictors_concat = sub_data['cond2_predictors']
        eeg_concat = sub_data['cond2_eeg']

    else:

        print(f"Skipping {sub_name}: no valid data")
        continue

    # z-score predictors column-wise
    # target envelope, distractor envelope, and button-press predictor
    predictors_concat = zscore(predictors_concat, axis=0)
    predictors_concat = np.nan_to_num(predictors_concat)

    # z-score EEG channel-wise
    eeg_concat = zscore(eeg_concat, axis=0)
    eeg_concat = np.nan_to_num(eeg_concat)

    # store concatenated and z-scored data
    subs_matrices[sub_name]['predictors'] = predictors_concat
    subs_matrices[sub_name]['eeg'] = eeg_concat
    subs_matrices[sub_name]['predictor_columns'] = predictor_columns

    print(sub_name)
    print("Predictors:", predictors_concat.shape)
    print("EEG:", eeg_concat.shape)


    # ========================================================
    # REGULARIZATION OPTIMIZATION PER SUBJECT
    # ========================================================

    X = predictors_concat
    Y = eeg_concat

    # split continuous data into chunks for cross-validation
    X_blocks = np.array_split(X, n_blocks, axis=0)
    Y_blocks = np.array_split(Y, n_blocks, axis=0)

    # use negative mean squared error
    # multiply by -1 to get the actual mean squared error
    trf_mse = TRF(metric=neg_mse)
    mse = trf_mse.train(
        X_blocks,
        Y_blocks,
        sfreq,
        tmin,
        tmax,
        lambdas
    ) * -1

    # use Pearson correlation as an additional performance metric
    trf_r = TRF(metric=pearsonr)
    r = trf_r.train(
        X_blocks,
        Y_blocks,
        sfreq,
        tmin,
        tmax,
        lambdas
    )

    # select best lambda based on minimum MSE
    best_lambda = lambdas[np.argmin(mse)]

    # save all best lambdas
    all_best_lambdas[sub_name] = best_lambda

    print(f"{sub_name} best lambda: {best_lambda}")


    # ========================================================
    # PLOT REGULARIZATION CURVES
    # ========================================================

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()

    ax1.semilogx(lambdas, r, color='c')
    ax2.semilogx(lambdas, mse, color='m')

    ax1.set(
        xlabel='Regularization value',
        ylabel='Correlation coefficient')

    ax2.set(
        ylabel='Mean squared error')

    ax1.axvline(
        best_lambda,
        linestyle='--',
        color='k')

    plt.title(f"{sub_name} regularization optimization")
    plt.show(block=False)
    plt.pause(0.001)


# ============================================================
# COMPUTE GROUP-LEVEL LAMBDA
# ============================================================

# use median in log-space because lambdas are logarithmically spaced
group_lambda = 10 ** np.median(np.log10(list(all_best_lambdas.values())))

print("Best lambdas per subject:")
print(all_best_lambdas)
print("Group lambda:", group_lambda)


# ============================================================
# RUN FINAL TRF WITH GROUP LAMBDA
# ============================================================

subs_predictions = {}

for sub_name, sub_data in subs_matrices.items():

    if 'predictors' not in sub_data or 'eeg' not in sub_data:
        print(f"Skipping {sub_name}: no final predictors/eeg found")
        continue

    print(f"Running final TRF for {sub_name}")

    X = sub_data['predictors']
    Y = sub_data['eeg']

    print("Predictors:", X.shape)
    print("EEG:", Y.shape)

    # predict
    trf = TRF(direction=1, method='ridge')  # forward model

    # use group lambda for individual subjects
    trf.train(
        stimulus=X,
        response=Y,
        fs=sfreq,
        tmin=tmin,
        tmax=tmax,
        regularization=group_lambda,
        seed=42)

    predictions, r = trf.predict(
        stimulus=X,
        response=Y,
        average=False)

    # average across channels → one TRF curve per predictor
    w0 = np.mean(trf.weights[0], axis=1)  # target envelope TRF weights
    w1 = np.mean(trf.weights[1], axis=1)  # distractor envelope TRF weights


    # ========================================================
    # SMOOTH TRF WEIGHTS
    # ========================================================

    # smooth out the TRF responses with the hamming window
    window_len = 11

    hamming_win = np.hamming(window_len)
    hamming_win /= hamming_win.sum()

    w0_smooth = np.convolve(w0, hamming_win, mode='same')
    w1_smooth = np.convolve(w1, hamming_win, mode='same')

    # time axis - always the same
    lags = trf.times


    # ========================================================
    # STORE SUBJECT RESULTS
    # ========================================================

    # temporary store in a dictionary where TRF results of all subs are saved
    subs_predictions[sub_name] = {
        'predictions': predictions,
        'r': r,
        'target_env': w0_smooth,
        'distractor_env': w1_smooth,
        'time': lags,
        'group_lambda': group_lambda,
        'subject_best_lambda': all_best_lambdas.get(sub_name, None),
        'predictor_columns': predictor_columns}


# ============================================================
# SAVE FINAL TRF RESULTS
# ============================================================

# save predicted weights:
np.savez(
    output_path / f'trf_results_{plane}.npz',
    data=subs_predictions)