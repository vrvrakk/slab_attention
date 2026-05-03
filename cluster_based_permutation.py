# loading and saving
import os
from pathlib import Path
# handling the data
import numpy as np
# plotting
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
# analysis
from mne.stats import permutation_cluster_1samp_test

default_path = Path.cwd()
output_path = default_path /'data'/'TRF'/'output'

planes = ['azimuth', 'elevation']
plane = planes[0]

# import TRF zip file:
trf_dict = np.load(output_path/f'trf_results_{plane}.npz', allow_pickle=True)
trf_dict = trf_dict['data'].item() # now zip file is operational and accessible

# run cluster-based non parametric permutation:

# collect subject-level TRFs
target_all = []
distractor_all = []

for sub_name in trf_dict.keys():
    # keys available: dict_keys(['predictions', 'r', 'target_env', 'distractor_env', 'time'])
    times = trf_dict[sub_name]['time']
    target_env = trf_dict[sub_name]['target_env']          # shape: (n_times,)
    distractor_env = trf_dict[sub_name]['distractor_env']  # shape: (n_times,)

    target_all.append(target_env)
    distractor_all.append(distractor_env)

target_all = np.array(target_all)          # shape: (n_subjects, n_times)
distractor_all = np.array(distractor_all)  # shape: (n_subjects, n_times)

# paired difference: target - distractor
diff_all = target_all - distractor_all     # shape: (n_subjects, n_times)

print("diff_all shape:", diff_all.shape) # it should be averaged across ROI channels & across time (1, time)

# comparison across subs now:

T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
    diff_all,
    n_permutations=5000,
    tail=0,          # two-sided test -> is target > distractor or distractor > target?
    threshold=None,  # automatic threshold
    seed=42)

# plot:
# get mean TRF responses across subject for target and distractor envelopes
target_mean = np.mean(target_all, axis=0)
distractor_mean = np.mean(distractor_all, axis=0)

# also get standard error of the mean -> how precise is the mean estimate across subs
target_sem = np.std(target_all, axis=0) / np.sqrt(target_all.shape[0])
distractor_sem = np.std(distractor_all, axis=0) / np.sqrt(distractor_all.shape[0])


plt.figure()

# plot the mean responses curve per stream
plt.plot(times, target_mean, label='Target')
plt.plot(times, distractor_mean, label='Distractor')

# plot the shaded error
plt.fill_between(times,
                 target_mean - target_sem,
                 target_mean + target_sem,
                 alpha=0.2)

plt.fill_between(times,
                 distractor_mean - distractor_sem,
                 distractor_mean + distractor_sem,
                 alpha=0.2)

# indicates beginning of stimulus
plt.axvline(0, linestyle='--')
# limits plot to 500ms post stimulus onset
plt.xlim(-0.1, 0.5)

# highlights any significant clusters that were detected (if)
for cluster, p_val in zip(clusters, cluster_p_values):
    if p_val < 0.05:
        inds = cluster[0]
        plt.axvspan(times[inds[0]], times[inds[-1]], alpha=0.2, color='gray')

plt.xlabel('Time lag (s)')
plt.ylabel('TRF weight (z-scored)')
plt.title('Group-average Envelope TRFs')
plt.legend()

plt.show()
