import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import xgboost as xgb
import seaborn as sns
import shap
import subprocess
import json
import os

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc,
    mean_squared_error
)
from scipy import stats
from caafe.preprocessing import make_datasets_numeric

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

st.set_page_config(layout="wide")
st.title("🚀 LLM Feature Engineering Dashboard")

# ─────────────────────────────────────────────
# DATASET SELECTION
# ─────────────────────────────────────────────

dataset = st.selectbox(
    "Select Dataset",
    [
        "breast-w", "heart", "blood", "car", "cmc",
        "credit-g", "eucalyptus", "pc1", "tic-tac-toe",
        "vehicle", "adult", "bank", "junglechess",
        "diabetes", "covtype", "myocardial", "communities",
        "arrhythmia", "bike", "crab", "housing",
        "insurance", "forest-fires", "wine"
    ]
)

file_name = f"./data/{dataset}.csv"

# ─────────────────────────────────────────────
# TASK DETECTION
# ─────────────────────────────────────────────

def detect_task_type(y):
    if np.issubdtype(y.dtype, np.number):
        return len(np.unique(y)) > 20
    return False

# ─────────────────────────────────────────────
# FEATURE GENERATION
# ─────────────────────────────────────────────

def generate_features(dataset, use_api=True):
    spec_path = f"./specs/specification_{dataset}.txt"
    log_path = f"./logs/{dataset}_gpt3.5"

    command = [
        "python", "main.py",
        "--problem_name", dataset,
        "--spec_path", spec_path,
        "--log_path", log_path
    ]

    if use_api:
        command.extend(["--use_api", "True", "--api_model", "llama-3.3-70b-versatile"])

    st.code(" ".join(command))

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        st.success("✅ Features generated")
    else:
        st.error("❌ Generation failed")
        st.text(result.stderr)

# ─────────────────────────────────────────────
# BASELINE
# ─────────────────────────────────────────────

def run_baseline():
    df = pd.read_csv(file_name)
    target = df.columns[-1]

    X = df.drop(target, axis=1)
    y = df[target].to_numpy()

    is_reg = detect_task_type(y)
    st.write("📌 Task:", "Regression" if is_reg else "Classification")

    le = preprocessing.LabelEncoder()

    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = le.fit_transform(X[col])

    if not is_reg:
        y = le.fit_transform(y)

    kf = KFold(5, shuffle=True, random_state=42) if is_reg else StratifiedKFold(5, shuffle=True, random_state=42)

    scores = []
    all_y_test, all_y_pred, all_prob = [], [], []

    for train_idx, test_idx in kf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        df_train, df_test = make_datasets_numeric(X_train, X_test, target)

        X_train = torch.nan_to_num(torch.tensor(df_train.to_numpy()))
        X_test  = torch.nan_to_num(torch.tensor(df_test.to_numpy()))

        model = xgb.XGBRegressor() if is_reg else xgb.XGBClassifier()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if is_reg:
            score = mean_squared_error(y_test, y_pred, squared=False)
        else:
            score = accuracy_score(y_test, y_pred)
            all_prob.extend(model.predict_proba(X_test)[:,1])

        scores.append(score)
        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)

    return {
        "scores": scores,
        "y_test": np.array(all_y_test),
        "y_pred": np.array(all_y_pred),
        "y_prob": np.array(all_prob),
        "model": model,
        "X_sample": X_test[:100],
        "is_reg": is_reg
    }

# ─────────────────────────────────────────────
# ENSEMBLE
# ─────────────────────────────────────────────

def run_ensemble():
    df = pd.read_csv(file_name)
    target = df.columns[-1]

    X = df.drop(target, axis=1)
    y = df[target].to_numpy()

    is_reg = detect_task_type(y)

    kf = KFold(5, shuffle=True, random_state=42) if is_reg else StratifiedKFold(5, shuffle=True, random_state=42)

    scores = []

    for train_idx, test_idx in kf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        preds = []

        for seed in [1,2,3]:
            model = xgb.XGBRegressor() if is_reg else xgb.XGBClassifier(random_state=seed)
            model.fit(X_train, y_train)
            preds.append(model.predict(X_test))

        preds = np.stack(preds)

        final = np.mean(preds, axis=0) if is_reg else stats.mode(preds, axis=0)[0]

        score = mean_squared_error(y_test, final, squared=False) if is_reg else accuracy_score(y_test, final)
        scores.append(score)

    return scores

# ─────────────────────────────────────────────
# LLMFE
# ─────────────────────────────────────────────

def run_llmfe():
    result = subprocess.run(["python", "evaluation.py"], capture_output=True, text=True)

    scores = []
    for line in result.stdout.split("\n"):
        if "Accuracy:" in line or "RMSE" in line:
            scores.append(float(line.split(":")[-1]))

    return scores

# ─────────────────────────────────────────────
# UI FLOW
# ─────────────────────────────────────────────

log_dir = f"./logs/{dataset}_gpt3.5"

st.subheader("⚙️ Feature Generation")

if os.path.exists(log_dir):
    st.success("✅ Features already exist")
else:
    if st.button("🚀 Generate Features"):
        with st.spinner("Generating..."):
            generate_features(dataset)

# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if os.path.exists(log_dir):

    if st.button("📊 Run Evaluation"):

        with st.spinner("Running models..."):
            baseline = run_baseline()
            ensemble_scores = run_ensemble()
            llmfe_scores = run_llmfe()

        baseline_scores = baseline["scores"]

        # METRICS
        st.subheader("📊 Comparison")

        col1, col2, col3 = st.columns(3)
        col1.metric("Baseline", np.mean(baseline_scores))
        col2.metric("Ensemble", np.mean(ensemble_scores))
        col3.metric("LLMFE", np.mean(llmfe_scores))

        # CHART
        st.bar_chart({
            "Baseline": baseline_scores,
            "Ensemble": ensemble_scores,
            "LLMFE": llmfe_scores
        })

        # CONFUSION MATRIX
        if not baseline["is_reg"]:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(baseline["y_test"], baseline["y_pred"])
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', ax=ax)
            st.pyplot(fig)

            # ROC
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(baseline["y_test"], baseline["y_prob"])
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr)
            st.pyplot(fig)

        # SHAP
        st.subheader("🔍 SHAP")
        explainer = shap.Explainer(baseline["model"])
        shap_values = explainer(baseline["X_sample"])

        fig = plt.figure()
        shap.summary_plot(shap_values, baseline["X_sample"], show=False)
        st.pyplot(fig)

        # DOWNLOAD
        df_res = pd.DataFrame({
            "Baseline": baseline_scores,
            "Ensemble": ensemble_scores,
            "LLMFE": llmfe_scores
        })

        st.download_button("Download CSV", df_res.to_csv(index=False), "results.csv")

        # SHOW FEATURES
        st.subheader("🧠 Generated Features")

        sample_dir = f"./logs/{dataset}_gpt3.5_split_1/samples"

        if os.path.exists(sample_dir):
            for file in os.listdir(sample_dir)[:3]:
                with open(os.path.join(sample_dir, file)) as f:
                    st.code(json.load(f).get("function", ""))