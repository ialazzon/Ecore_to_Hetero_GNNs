"""
Random Forest baseline for multi-class FloorState classification, built on
flattened_data.csv (the per-sensor CSV from flatten_sensors_to_csv.py).

Split protocol: uses the EXACT train/val/test snapshot assignment recorded
in SPLITS_TXT (exp_splits.txt), rather than re-splitting itself. That file
was written out by the GNN training script as the ground-truth record of
which snapshot went into which split, so this guarantees the RF baseline is
evaluated on the identical rows as train_graph_clf_multiclass.py /
train_graph_clf_multiclass_basic.py -- same snapshots in train/val/test, no
re-randomization, no risk of drift between scripts.

Feature engineering (the aggregation step deliberately left OUT of the
flattening script) happens here: flattened_data.csv has one row per sensor,
but a snapshot can have multiple sensors of the same modality across
different rooms, and different snapshots sample different subsets of the 6
modalities. So per snapshot, for each modality, we average that modality's
readings across however many sensors of it exist, and also keep a COUNT
(some snapshots simply don't have a given modality at all -- that absence is
itself information, not just a missing value to impute away).

Validation set: RF doesn't need it for training (no epochs, no early
stopping) -- it's used here only for a small hyperparameter search
(n_estimators / max_depth), then the selected model's TEST performance is
reported, exactly mirroring how the GNN scripts used val to pick the best
checkpoint before touching test once.
"""

import ast
import re

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

DATA_CSV = 'flattened_data.csv'
SPLITS_TXT = 'exp_splits.txt'   # ground-truth train/val/test snapshot assignment
SEED = 0
METRICS = ['mean', 'max', 'slope', 'variance', 'zScoreVsHistory']

# --------------------------------------------------------------------------- #
# 1. Load + aggregate per-sensor rows into one feature vector per snapshot
# --------------------------------------------------------------------------- #

df = pd.read_csv(DATA_CSV)
print(f'Loaded {len(df)} sensor rows from {df.snapshot_id.nunique()} snapshots.')

# average each modality's readings across all sensors of that modality within
# a snapshot (there can be more than one, e.g. two CO2 sensors in different
# rooms) -> columns like CO2_mean, CO2_max, ..., Temperature_zScoreVsHistory
agg = df.groupby(['snapshot_id', 'modality'])[METRICS].mean().unstack('modality')
agg.columns = [f'{modality}_{metric}' for metric, modality in agg.columns]

# how many sensors of each modality this snapshot has -- absence of a
# modality is itself informative, not just a gap to fill blindly
counts = df.groupby(['snapshot_id', 'modality']).size().unstack('modality')
counts.columns = [f'{modality}_count' for modality in counts.columns]

X = pd.concat([agg, counts], axis=1).fillna(0.0)
y = df.groupby('snapshot_id')['label'].first().loc[X.index]

print(f'Built {X.shape[0]} snapshot feature vectors, {X.shape[1]} features each.')
print('Feature columns:', list(X.columns))

CLASSES = sorted(y.unique())
print(f'\nDiscovered {len(CLASSES)} classes: {CLASSES}')


# --------------------------------------------------------------------------- #
# 2. Load the fixed train/val/test split from SPLITS_TXT
# --------------------------------------------------------------------------- #

def load_splits(splits_txt):
    """Parse a file with lines like:
        Training  (417): ['snapshot_0000', 'snapshot_0001', ...]
        Test      (96): [...]
        Validation(87): [...]
    into a dict {'Training': [...], 'Test': [...], 'Validation': [...]}.
    Matching on the leading word is case-insensitive and tolerant of the
    count/whitespace formatting so this keeps working if the GNN script's
    print formatting shifts slightly.
    """
    with open(splits_txt) as f:
        text = f.read()

    splits = {}
    for name in ('Training', 'Test', 'Validation'):
        m = re.search(rf'{name}\s*\(\d+\)\s*:\s*(\[.*?\])', text)
        if not m:
            raise RuntimeError(f'Could not find a "{name}" split line in {splits_txt}')
        splits[name] = ast.literal_eval(m.group(1))
    return splits


splits = load_splits(SPLITS_TXT)
train_names, test_names, val_names = splits['Training'], splits['Test'], splits['Validation']
print(f'\nLoaded splits from {SPLITS_TXT}: '
      f'train {len(train_names)}, val {len(val_names)}, test {len(test_names)}')


def _select(names, split_label):
    names_in_data = [n for n in names if n in X.index]
    missing = [n for n in names if n not in X.index]
    if missing:
        print(f'  [warn] {split_label}: {len(missing)} snapshot(s) from '
              f'{SPLITS_TXT} not found in {DATA_CSV}, skipping them '
              f'(e.g. {missing[:5]})')
    return X.loc[names_in_data], y.loc[names_in_data]


X_train, y_train = _select(train_names, 'Training')
X_val, y_val = _select(val_names, 'Validation')
X_test, y_test = _select(test_names, 'Test')

print(f'train: {len(X_train)}  val: {len(X_val)}  test: {len(X_test)}')
print('class balance (overall):', dict(y.value_counts().sort_index()))
print('class balance (train):', dict(y_train.value_counts().sort_index()))
print('class balance (val):  ', dict(y_val.value_counts().sort_index()))
print('class balance (test): ', dict(y_test.value_counts().sort_index()))


# --------------------------------------------------------------------------- #
# 3. Small hyperparameter search on val (RF's equivalent of the GNN's
#    best-val-checkpoint selection -- no iterative training here, so this is
#    the one place val actually earns its keep)
# --------------------------------------------------------------------------- #

param_grid = [
    dict(n_estimators=n, max_depth=d)
    for n in (100, 300, 500)
    for d in (None, 10, 20)
]

best_val_acc = -1.0
best_model = None
best_params = None

for params in param_grid:
    model = RandomForestClassifier(random_state=SEED, n_jobs=-1, **params)
    model.fit(X_train, y_train)
    val_acc = accuracy_score(y_val, model.predict(X_val))
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model = model
        best_params = params

print(f'\nBest hyperparameters (by val acc): {best_params}')
print(f'Best Val Acc: {best_val_acc*100:.2f}%')


# --------------------------------------------------------------------------- #
# 4. Final test evaluation (touched once, using the val-selected model)
# --------------------------------------------------------------------------- #

y_pred = best_model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_pred, labels=CLASSES, zero_division=0)
macro_f1 = f1.mean().item()

print('\n================ RESULT (Random Forest, tabular baseline) ================')
print(f'Test Acc: {test_acc*100:.2f}%\n')
print(f'Macro-F1: {macro_f1:.4f}\n')
print(f'{"class":<16}{"precision":>10}{"recall":>10}{"f1":>10}{"support":>10}')
for c, p, r, f, s in zip(CLASSES, precision, recall, f1, support):
    print(f'{c:<16}{p:>10.3f}{r:>10.3f}{f:>10.3f}{int(s):>10}')

# Feature importance -- useful sanity check on what's actually driving the RF
importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print('\nTop 10 most important features:')
print(importances.head(10).to_string())
