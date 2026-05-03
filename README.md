**1. EEG Preprocessing**
Script: 1_preprocessing.py

**Steps:**
Load raw EEG
Set electrode montage (standard 10–20)
Visual inspection:
mark bad channels
mark bad segments
Interpolate bad channels
Remove line noise (50 Hz)
Bandpass filter (1–30 Hz)
Concatenate blocks
Apply ICA (remove eye artifacts)
Re-reference (average)
Downsample to 125 Hz
Save cleaned EEG

**2. Create Auxiliary Predictors**

Script: 2_bad_segments_and_responses.py

**Outputs:**
A) Bad segment array
0 = good data
-999 = bad segment
→ used to remove or zero-out bad EEG periods

B) Button press array
0 = no response
1 = button press
→ used as a predictor in TRF

**3️. Create Envelope Predictors**
Script: 3_envelopes.py

**Steps:**
Load stimulus envelopes
Map stimuli numbers to correct sounds
Separate:
target stream
distractor stream
Insert envelopes at correct onset times
Concatenate across blocks
Output:
target_envelope_concat
distractor_envelope_concat

**4️. Build TRF Input Matrix**
Script: 4_build_trf_matrix.py

**Combines:**
EEG (ROI channels)
target envelope
distractor envelope
button presses
Handles bad segments:

**Either:**

X remove them
or
set them to zero (preferred in this pipeline)
Final matrices:
X = [target_env, distractor_env, button_press]
Y = EEG

**Shapes:**

X → (samples, 3 predictors)
Y → (samples, channels)

**Saved as:**

subXX_condition_matrix.npz

**5️.TRF Model**

**Script:** 5_TRF.py

**Steps:
**Combine conditions

**Example:**

azimuth = a1 + a2
elevation = e1 + e2

_Z-scoring_
predictors → column-wise
EEG → per channel
**Regularization optimization**
**For each subject:**

split data into 5 blocks
test multiple λ values (ridge regression)
select best λ (min MSE)
**Group λ**
group_lambda = median(log(lambdas))

Used to standardize models across subjects.

**Final TRF**
fit model using group λ
extract TRF weights:
target
distractor
smooth with Hamming window
_Output:_

**Saved per subject:**

{
    'target_env': TRF curve
    'distractor_env': TRF curve
    'time': lags
}
**6. Statistical Analysis**

Script: 6_cluster_based_permutation.py

**Goal: Test whether:**

target TRF ≠ distractor TRF
**Method:**

Cluster-based permutation test

**Using:**

permutation_cluster_1samp_test
Steps:
Collect TRFs across subjects

**Compute difference:**

diff = target - distractor
Test against 0 across time
Output:
significant time windows
plotted TRF curves
shaded significant clusters

**Final Visualization**

**Plot shows:**
mean target TRF
mean distractor TRF
SEM (shaded)
significant clusters (gray)

**Key Concepts**: TRF (Temporal Response Function)

**Maps:**
stimulus → brain response over time
Regularization (λ)

**Controls:**

smoothness of TRF
prevents overfitting
Cluster permutation test

**Finds:**
time windows where responses differ
controls for multiple comparisons

**Important Notes**
You need multiple subjects for statistics
With 1 subject:
no variance
no valid cluster test
Bad segment handling:
removing samples → cleaner
zeroing → preserves timing
both are acceptable
Z-scoring is essential:
makes subjects comparable
stabilizes regression

**Summary**
**Pipeline flow:**

RAW EEG -> Preprocessing -> Predictors (envelopes + responses) -> TRF input matrix ->
TRF model (ridge regression) -> Group-level lambda -> Final TRFs -> Cluster permutation test
