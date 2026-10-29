"""Train, compare and save the churn classification pipeline.

Flow: load -> validate -> clean -> split -> preprocess (ColumnTransformer)
      -> train 3 models -> evaluate -> pick best by real metrics -> save.
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = "data/customer_churn.csv"
MODEL_PATH = "model/churn_model.pkl"
METRICS_PATH = "model/metrics.json"
RANDOM_STATE = 42

NUMERIC_COLS = ["Age", "TenureMonths", "MonthlyCharges", "WatchHoursPerWeek",
                "LoginFrequencyPerWeek", "PaymentFailures", "CustomerComplaints"]
CATEGORICAL_COLS = ["Gender", "SubscriptionPlan", "AutoRenewal"]
TARGET = "Churn"
REQUIRED_COLS = ["CustomerID"] + NUMERIC_COLS + CATEGORICAL_COLS + [TARGET]


def load_dataset() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        print(f"{DATA_PATH} not found -> generating SYNTHETIC dataset.")
        from generate_data import generate
        os.makedirs("data", exist_ok=True)
        generate().to_csv(DATA_PATH, index=False)
    return pd.read_csv(DATA_PATH)


def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(f"Dataset is missing required columns: {missing}")

    df = df.drop_duplicates(subset="CustomerID").copy()
    df[TARGET] = df[TARGET].astype(str).str.strip().str.capitalize()
    df = df[df[TARGET].isin(["Yes", "No"])]
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_COLS:
        df[col] = df[col].apply(lambda v: v.strip() if isinstance(v, str) else np.nan)
    return df.reset_index(drop=True)


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric, NUMERIC_COLS),
                              ("cat", categorical, CATEGORICAL_COLS)])


def main():
    df = validate_and_clean(load_dataset())
    print(f"Rows after cleaning: {len(df)}")
    print(df[TARGET].value_counts(normalize=True).round(3).to_dict())

    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = (df[TARGET] == "Yes").astype(int)          # 1 = churn

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20,
                                                random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                random_state=RANDOM_STATE, n_jobs=-1),
    }

    results, fitted = {}, {}
    for name, clf in candidates.items():
        pipe = Pipeline([("preprocessor", build_preprocessor()), ("classifier", clf)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        results[name] = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1_score": f1_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, proba),
        }
        fitted[name] = pipe
        print(f"\n{name}: " + ", ".join(f"{k}={v:.3f}" for k, v in results[name].items()))

    # Select on actual results: ROC-AUC first, F1 as tie-breaker
    best_name = max(results, key=lambda n: (round(results[n]["roc_auc"], 4),
                                            results[n]["f1_score"]))
    print(f"\nSelected model: {best_name}")

    os.makedirs("model", exist_ok=True)
    joblib.dump(fitted[best_name], MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump({
            "selected_model": best_name,
            "selection_rule": "highest ROC-AUC on the held-out test set (F1 breaks ties)",
            "metrics": results[best_name],
            "all_models": results,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "feature_columns": NUMERIC_COLS + CATEGORICAL_COLS,
        }, f, indent=2)
    print(f"Saved {MODEL_PATH} and {METRICS_PATH}")


if __name__ == "__main__":
    main()
