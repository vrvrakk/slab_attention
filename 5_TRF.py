import os
from pathlib import Path
import numpy as np
import mne
from mtrf import TRF
from mtrf.stats import crossval, neg_mse, pearsonr
import pandas as pd
# plotting
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
plt.ion()

# define your paths where you will load/save files
default_path = Path.cwd()
data_path = default_path / 'data'
input_path = data_path / 'TRF' / 'input'
output_path = data_path / 'TRF' / 'output'
output_path.mkdir(parents=True, exist_ok=True)


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

    elif conditions[1] in files.name:
        cond2_matrix = np.load(files, allow_pickle=True)
        # zscore across each column of predictors
        subs_matrices[sub_name]['cond2_predictors'] = cond2_matrix['predictors']
        # zscore across channels
        subs_matrices[sub_name]['cond2_eeg'] = cond2_matrix['eeg']


# concatenate cond1 + cond2 per subject and zscore
all_best_lambdas = {}
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

    # regularization optimization:
    n_blocks = 5
    X_blocks = np.array_split(X, n_blocks, axis=0)
    Y_blocks = np.array_split(Y, n_blocks, axis=0)

    lambdas = np.logspace(-2, 2, 20)  # based on prev literature
    tmin, tmax = -0.1, 1.0
    trf = TRF(metric=neg_mse)  # use negative meas squared error
    # multiply by -1 to get the mean squared error
    mse = trf.train(X_blocks, Y_blocks, sfreq, tmin, tmax, lambdas) * -1

    trf = TRF(metric=pearsonr)  # use pearsons correlation
    r = trf.train(X_blocks, Y_blocks, sfreq, tmin, tmax, lambdas)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.semilogx(lambdas, r, color='c')
    ax2.semilogx(lambdas, mse, color='m')
    ax1.set(xlabel='Regularization value', ylabel='Correlation coefficient')
    ax2.set(ylabel='Mean squared error')
    ax1.axvline(lambdas[np.argmin(mse)], linestyle='--', color='k')
    plt.show()

    best_lambda = lambdas[np.argmin(mse)]

    # save all best lambdas:
    all_best_lambdas[sub_name] = best_lambda
##################################################
    'RUN TRF WITH GROUP LAMBDA'
##################################################
subs_predictions = {}
group_lambda = 10 ** np.median(np.log10(list(all_best_lambdas.values())))

for sub_name in subs_matrices.keys():
    subs_matrices[sub_name]['predictors'] = predictors_concat
    subs_matrices[sub_name]['eeg'] = eeg_concat
    subs_matrices[sub_name]['predictor_columns'] = predictor_columns

    print(sub_name)
    print("Predictors:", predictors_concat.shape)
    print("EEG:", eeg_concat.shape)
    X = predictors_concat
    Y = eeg_concat

    # predict
    trf = TRF(direction=1, method='ridge')  # forward model
    # use group lambda for individual subjects
    trf.train(stimulus=X, response=Y, fs=sfreq, tmin=-0.1, tmax=1.0, regularization=group_lambda, seed=42)
    predictions, r = trf.predict(stimulus=X, response=Y, average=False)

    # average across channels → (lags,)
    w0 = np.mean(trf.weights[0], axis=1) # target envelope TRF weights
    w1 = np.mean(trf.weights[1], axis=1) # distractor envelope TRF weights

    # --- smoothing ---
    # smooth out the TRF responses with the hamming window
    window_len = 11
    hamming_win = np.hamming(window_len)
    hamming_win /= hamming_win.sum()

    w0_smooth = np.convolve(w0, hamming_win, mode='same')
    w1_smooth = np.convolve(w1, hamming_win, mode='same')

    # time axis - always the same
    lags = trf.times

    # temporary store in a dictionary where TRF results of all subs are saved
    subs_predictions[sub_name] = {'predictions': predictions, 'r': r, 'target_env': w0_smooth, 'distractor_env': w1_smooth, 'time': lags}

# save predicted weights:
np.savez(output_path / f'trf_results_{plane}.npz', data=subs_predictions)









