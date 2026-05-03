# ============================================================
# CLUSTER-BASED PERMUTATION TEST FOR TRF RESULTS
#
# This script:
# 1. Loads saved TRF results for one plane
# 2. Extracts target and distractor TRF curves for all subjects
# 3. Computes paired target - distractor differences
# 4. Runs a two-sided cluster-based permutation test
# 5. Plots group-average target/distractor TRFs
# 6. Highlights significant time clusters
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

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


# ============================================================
# PATHS
# ============================================================

default_path = Path.cwd()
output_path = default_path / 'data' / 'TRF' / 'output'


# ============================================================
# USER SETTINGS
# ============================================================

planes = ['azimuth', 'elevation']
plane = planes[0]

n_permutations = 5000
alpha = 0.05

plot_tmin = -0.1
plot_tmax = 0.5


# ============================================================
# LOAD TRF RESULTS
# ============================================================

# import TRF zip file:
trf_dict = np.load(
    output_path / f'trf_results_{plane}.npz',
    allow_pickle=True)

trf_dict = trf_dict['data'].item()  # now zip file is operational and accessible


# ============================================================
# COLLECT SUBJECT-LEVEL TRFS
# ============================================================

# run cluster-based non parametric permutation:

# collect subject-level TRFs
target_all = []
distractor_all = []

for sub_name in trf_dict.keys():

    # keys available:
    # dict_keys(['predictions', 'r', 'target_env', 'distractor_env', 'time'])

    times = trf_dict[sub_name]['time']

    target_env = trf_dict[sub_name]['target_env']          # shape: (n_times,)
    distractor_env = trf_dict[sub_name]['distractor_env']  # shape: (n_times,)

    target_all.append(target_env)
    distractor_all.append(distractor_env)

target_all = np.array(target_all)          # shape: (n_subjects, n_times)
distractor_all = np.array(distractor_all)  # shape: (n_subjects, n_times)

print("target_all shape:", target_all.shape)
print("distractor_all shape:", distractor_all.shape)


# ============================================================
# COMPUTE TARGET - DISTRACTOR DIFFERENCE
# ============================================================

# paired difference: target - distractor
# this is equivalent to a paired comparison between the two curves
diff_all = target_all - distractor_all     # shape: (n_subjects, n_times)

print("diff_all shape:", diff_all.shape)
# should be: n_subjects x n_times
# each subject already has one ROI-averaged TRF curve over time


# ============================================================
# RUN CLUSTER-BASED PERMUTATION TEST
# ============================================================

# comparison across subs now:
# one-sample cluster test on target - distractor differences
# tests whether the difference is significantly different from zero over time

T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
    diff_all,
    n_permutations=n_permutations,
    tail=0,          # two-sided test -> is target > distractor OR distractor > target?
    threshold=None,  # automatic threshold
    seed=42)


# ============================================================
# PRINT SIGNIFICANT CLUSTERS
# ============================================================

sig_clusters = []

for cluster_idx, (cluster, p_val) in enumerate(zip(clusters, cluster_p_values)):

    if p_val < alpha:

        inds = cluster[0]

        cluster_start = times[inds[0]]
        cluster_end = times[inds[-1]]

        sig_clusters.append((cluster_idx, cluster_start, cluster_end, p_val))

        print(
            f"Significant cluster {cluster_idx}: "
            f"{cluster_start:.3f}–{cluster_end:.3f} s, "
            f"p = {p_val:.4f}")

if len(sig_clusters) == 0:
    print("No significant clusters found.")


# ============================================================
# COMPUTE GROUP-AVERAGE TRFS
# ============================================================

# get mean TRF responses across subjects for target and distractor envelopes
target_mean = np.mean(target_all, axis=0)
distractor_mean = np.mean(distractor_all, axis=0)

# also get standard error of the mean -> how precise is the mean estimate across subs
target_sem = np.std(target_all, axis=0) / np.sqrt(target_all.shape[0])
distractor_sem = np.std(distractor_all, axis=0) / np.sqrt(distractor_all.shape[0])


# ============================================================
# PLOT GROUP-AVERAGE TARGET AND DISTRACTOR TRFS
# ============================================================

plt.figure()

# plot the mean responses curve per stream
plt.plot(times, target_mean, color='blue', label='Target')
plt.plot(times, distractor_mean, color='r', label='Distractor')

# plot the shaded error
plt.fill_between(
    times,
    target_mean - target_sem,
    target_mean + target_sem,
    alpha=0.2)

plt.fill_between(
    times,
    distractor_mean - distractor_sem,
    distractor_mean + distractor_sem,
    alpha=0.2)

# indicates beginning of stimulus
plt.axvline(0, linestyle='--')

# limits plot to 500ms post stimulus onset
plt.xlim(plot_tmin, plot_tmax)

# highlights any significant clusters that were detected (if)
for cluster, p_val in zip(clusters, cluster_p_values):

    if p_val < alpha:

        inds = cluster[0]

        plt.axvspan(
            times[inds[0]],
            times[inds[-1]],
            alpha=0.2,
            color='gray')

plt.xlabel('Time lag (s)')
plt.ylabel('TRF weight (z-scored)')
plt.title(f'Group-average Envelope TRFs: {plane}')
plt.legend()

plt.show(block=False)

# ============================================================
# SAVE FIGURE AS PDF / JPEG
# ============================================================

fig_save_path = output_path / 'figures'
fig_save_path.mkdir(parents=True, exist_ok=True)

plt.savefig(
    fig_save_path / f'group_average_envelope_trf_{plane}.pdf',
    bbox_inches='tight')

plt.savefig(
    fig_save_path / f'group_average_envelope_trf_{plane}.jpg',
    dpi=300,
    bbox_inches='tight')