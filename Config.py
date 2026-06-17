"""
Config.py
=========
Shared constants for the MCL experiments (tasks, feature groups, signal and
training settings). Import these instead of hard-coding values in each script.
"""
import os

SEED = 42
DATA_ROOT = os.environ.get("MCL_DATA", "./data")

# tasks -> number of classes
TASKS = {"nback": 3, "dsr": 2, "wg": 2}
TASK_LABELS = {
    "nback": ["0-back", "2-back", "3-back"],
    "dsr":   ["target", "non-target"],
    "wg":    ["word-generation", "baseline"],
}

# feature groups; FDS and TDT are the two carried to the cross-subject stage
FEATURE_GROUPS = ("TDT", "TDS", "TDM", "FDS", "FDP", "TFD")
BEST_GROUPS = ("FDS", "TDT")

# signals / preprocessing
EEG_FS, NIRS_FS = 200, 10
EEG_BAND = (1.0, 45.0)
NIRS_BAND = (0.01, 0.2)
WIN_S, OVERLAP_S = 3.0, 1.0
EEG_SHAPE = (18, 360, 1)
NIRS_SHAPE = (18, 72, 1)

# training (matches ModelInitialization.py / TrainingLoop.py)
LR = 1e-4
META_ITERATIONS = 100
INNER_STEPS = 5
NUM_TASKS = 10
BASELINE_EPOCHS = 150
BATCH_SIZE = 32
TEST_SIZE = 0.2

# constants for the asymptotic complexity expression
COMPLEXITY = dict(N_t=NUM_TASKS, S_i=INNER_STEPS, T_e=180, F_e=40,
                  T_f=60, F_f=20, H=64)
