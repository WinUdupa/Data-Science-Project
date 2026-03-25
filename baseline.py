"""
Baseline evaluation script (consistent with evaluation.py, without LLM features)
"""

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from caafe.preprocessing import make_datasets_numeric

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

file_name = "./data/heart.csv"
seed = 42

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

df = pd.read_csv(file_name)
target_attr = df.columns[-1]

# Separate features and target
X = df.drop(target_attr, axis=1)
y = df[target_attr].to_numpy()

# Encode categorical features
label_encoder = preprocessing.LabelEncoder()

for col in X.columns:
    if X[col].dtype == 'object' or X[col].dtype == 'string':
        X[col] = label_encoder.fit_transform(X[col])

# Encode target (classification)
y = label_encoder.fit_transform(y)

# ─────────────────────────────────────────────
# CROSS-VALIDATION
# ─────────────────────────────────────────────

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

scores = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
    print(f"\n--- Fold {fold} ---")

    X_train_fold = X.iloc[train_idx]
    X_test_fold  = X.iloc[test_idx]
    y_train_fold = y[train_idx]
    y_test_fold  = y[test_idx]

    # SAME preprocessing as evaluation.py
    df_train_new, df_test_new = make_datasets_numeric(
        X_train_fold.copy(),
        X_test_fold.copy(),
        target_attr
    )

    # Convert to tensors
    X_train_tensor = torch.tensor(df_train_new.to_numpy())
    X_test_tensor  = torch.tensor(df_test_new.to_numpy())

    # Handle NaNs (same as evaluation.py)
    X_train_tensor = torch.nan_to_num(X_train_tensor)
    X_test_tensor  = torch.nan_to_num(X_test_tensor)

    # Model (same as evaluation.py)
    model = xgb.XGBClassifier(random_state=42)

    model.fit(X_train_tensor, y_train_fold)
    y_pred = model.predict(X_test_tensor)

    acc = accuracy_score(y_test_fold, y_pred)
    print(f"  Fold {fold} Accuracy: {acc:.4f}")

    scores.append(acc)

# ─────────────────────────────────────────────
# FINAL RESULT
# ─────────────────────────────────────────────

mean_acc = np.mean(scores)
std_acc = np.std(scores)

print("\n>>> Baseline (XGBoost - consistent): {:.3f} ± {:.3f}".format(mean_acc, std_acc))