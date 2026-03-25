"""
Evaluation script for LLMFE generated features.
Converted from evaluation.ipynb
"""

import os
import json
import heapq
import statistics
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from scipy import stats
from caafe.preprocessing import make_datasets_numeric
from optimization_utils import is_categorical


# ─────────────────────────────────────────────
# CONFIGURATION — change these as needed
# ─────────────────────────────────────────────

# Add or remove dataset names here
pb_name = ['heart']         # e.g. ['heart', 'breast-w', 'crab']
llm     = "gpt3.5"          # used to locate the log folder
seed    = 42

# ─────────────────────────────────────────────
# MAIN EVALUATION LOOP
# ─────────────────────────────────────────────

label_encoder = preprocessing.LabelEncoder()

for problem_name in pb_name:
    print(f"\n{'='*50}")
    print(f"Evaluating: {problem_name}")
    print(f"{'='*50}")

    # Determine task type
    is_regression = problem_name in ['forest-fires', 'housing', 'insurance', 'bike', 'wine', 'crab']

    # Load dataset
    file_name = f"./data/{problem_name}.csv"
    df = pd.read_csv(file_name)
    target_attr = df.columns[-1]

    is_cat = [is_categorical(df.iloc[:, i]) for i in range(df.shape[1])][:-1]
    attribute_names = df.columns[:-1].tolist()

    X = df.convert_dtypes()
    y = df[target_attr].to_numpy()

    X = X.drop(target_attr, axis=1)
    for col in X.columns:
        if X[col].dtype == 'string':
            X[col] = label_encoder.fit_transform(X[col])

    if not is_regression:
        y = label_encoder.fit_transform(y)

    # Cross-validation split
    if is_regression:
        skf = KFold(n_splits=5, shuffle=True, random_state=seed)
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    max_score_avg = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n--- Fold {fold_idx} ---")

        label_encoder = preprocessing.LabelEncoder()
        X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]

        # Load generated feature samples from logs
        directory = f'./logs/{problem_name}_{llm}_split_{fold_idx}/samples'

        if not os.path.exists(directory):
            print(f"Log directory not found: {directory} — skipping fold.")
            continue

        # Read scores from all sample JSON files
        scores = []
        max_score = -1000000.0

        for idx, filename in enumerate(os.listdir(directory)):
            if filename.endswith(".json"):
                filepath = os.path.join(directory, filename)
                with open(filepath, "r") as f:
                    if idx == 1:
                        scores.append(max_score)
                    scores.append(json.load(f)['score'])

        scores = [max_score if v is None else v for v in scores]

        # Pick top 3 scoring feature functions
        best_k = heapq.nlargest(3, range(len(scores)), key=scores.__getitem__)

        test_outputs_all = []

        for filename in best_k:
            if filename == 1:
                continue
            filepath = f'{directory}/samples_{filename}.json'
            with open(filepath, "r") as f:
                feature_fn = json.load(f)['function']

            try:
                # Execute the generated feature function
                exec(feature_fn, globals())

                df_train_modified = modify_features(X_train_fold)
                for col in df_train_modified.columns:
                    if df_train_modified[col].dtype == 'string':
                        df_train_modified[col] = label_encoder.fit_transform(df_train_modified[col])

                df_test_modified = modify_features(X_test_fold)
                for col in df_test_modified.columns:
                    if df_test_modified[col].dtype == 'string':
                        df_test_modified[col] = label_encoder.fit_transform(df_test_modified[col])

                df_train_new, df_test_new = make_datasets_numeric(df_train_modified, df_test_modified, target_attr)

                X_train = torch.tensor(df_train_new.to_numpy())
                X_test  = torch.tensor(df_test_new.to_numpy())

                if is_regression:
                    model   = xgb.XGBRegressor(random_state=42)
                    y_train = y_train_fold
                    y_test  = y_test_fold
                else:
                    model   = xgb.XGBClassifier(random_state=42)
                    y_train = label_encoder.fit_transform(y_train_fold)
                    y_test  = label_encoder.fit_transform(y_test_fold)

                model.fit(torch.nan_to_num(X_train), y_train)
                y_pred = model.predict(torch.nan_to_num(X_test))
                test_outputs_all.append(y_pred)

            except Exception as e:
                print(f"  Skipping sample {filename} due to error: {e}")
                continue

        # Aggregate predictions across top-k features
        if len(test_outputs_all) == 0:
            print("  No valid predictions for this fold.")
            continue

        test_outputs_all = np.stack(test_outputs_all, axis=0)

        if is_regression:
            pred  = np.mean(test_outputs_all, axis=0)
            score = -mean_squared_error(y_test, pred, squared=False)
            print(f"  Fold {fold_idx} RMSE score: {score:.4f}")
        else:
            pred  = stats.mode(test_outputs_all, axis=0)[0]
            score = accuracy_score(y_test, pred)
            print(f"  Fold {fold_idx} Accuracy: {score:.4f}")

        max_score_avg.append(score)

    # Final summary for this dataset
    if len(max_score_avg) > 0:
        mean_val = sum(max_score_avg) / len(max_score_avg)
        std_dev  = statistics.stdev(max_score_avg) if len(max_score_avg) > 1 else 0.0
        print(f"\n>>> {problem_name} : {mean_val:.3f} +- {std_dev:.3f}")
    else:
        print(f"\n>>> {problem_name} : No valid results found.")