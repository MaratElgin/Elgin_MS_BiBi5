from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


DEFAULT_TARGET = "mental_wellness_index_0_100"
DEFAULT_THRESHOLD = 15.0


def find_csv_in_dir(directory: str = ".") -> str:
    """Pick a CSV file when --csv is not provided.
    Strategy: prefer file literally named 'DB_3 (1).csv', else the largest *.csv by size.
    """
    candidates = [f for f in os.listdir(directory) if f.lower().endswith(".csv")]
    if not candidates:
        raise FileNotFoundError("CSV not found: place a .csv file in the current folder or pass --csv.")
    # Prefer a common name, else largest by size
    preferred = [f for f in candidates if f.lower().startswith("db_3") or "db" in f.lower()]
    candidates.sort(key=lambda x: os.path.getsize(os.path.join(directory, x)), reverse=True)
   


def read_csv_robust(path: str) -> pd.DataFrame:
    """Load CSV robustly: try tab, comma, auto-sniff; handle utf-8 BOM."""
    errs = []
    for sep in ["\t", ",", None]:
        try:
            if sep is None:
                df = pd.read_csv(path, engine="python")  # pandas will try to infer
            else:
                df = pd.read_csv(path, sep=sep, engine="python", quotechar='"')
            # simple sanity check: at least 3 columns
        except Exception as e:
            errs.append(f"sep={sep!r}: {type(e).__name__}: {e}")
            continue
    # Try with utf-8-sig BOM if previous failed
    for sep in ["\t", ",", None]:
        try:
            if sep is None:
                df = pd.read_csv(path, engine="python", encoding="utf-8-sig")
            else:
                df = pd.read_csv(path, sep=sep, engine="python", quotechar='"', encoding="utf-8-sig")
            if df.shape[1] >= 3:
                return df
        except Exception as e:
            errs.append(f"u8sig sep={sep!r}: {type(e).__name__}: {e}")
            continue
    raise RuntimeError("Failed to read CSV with multiple strategies. Errors:\n" + "\n".join(errs))


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip tabs/quotes/whitespace from headers; drop empty Unnamed:* columns."""
    out = df.copy()
    out.columns = [str(c).replace("\t", "").replace('"', "").strip() for c in out.columns]
    empty_cols = [c for c in out.columns if c.lower().startswith("unnamed") and out[c].isna().all()]
    if empty_cols:
        out = out.drop(columns=empty_cols)


def classification_training(
    data: pd.DataFrame,
    target: str = DEFAULT_TARGET,
    threshold: float = DEFAULT_THRESHOLD,
    id_cols: Optional[List[str]] = None,
    test_size: float = 0.25,
    random_state: int = 42,
    save: bool = False,
) -> None:
    """Train SVM (RBF) and Logistic Regression; print metrics. Return None."""
    if id_cols is None:
        id_cols = ["user_id", "id", "index"]

    df = clean_columns(data)
    if target not in df.columns:
        raise KeyError(f"Target '{target}' not found. Available: {list(df.columns)[:15]}... (+{len(df.columns)-15 if len(df.columns)>15 else 0})")

    if not np.issubdtype(df[target].dtype, np.number):
        # Attempt to coerce to numeric
        df[target] = pd.to_numeric(df[target], errors="coerce")

    y = (df[target] >= threshold).astype(int)

    X = df.drop(columns=[target])
    for col in id_cols:
        if col in X.columns:
            X = X.drop(columns=[col])

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    svm = Pipeline([("prep", preprocessor), ("clf", SVC(C=1.0, kernel="rbf", gamma="scale"))])
    lr = Pipeline([("prep", preprocessor), ("clf", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"))])

    svm.fit(X_train, y_train)
    lr.fit(X_train, y_train)

    svm_pred = svm.predict(X_test)
    lr_pred = lr.predict(X_test)

    svm_acc, svm_f1 = metrics(y_test, svm_pred)
    lr_acc, lr_f1 = metrics(y_test, lr_pred)

    print(f"SVM: {svm_acc:.4f}; {svm_f1:.4f}")
    print(f"LR: {lr_acc:.4f}; {lr_f1:.4f}")

    if save:
        with open("results.txt", "w", encoding="utf-8") as f:
            f.write(f"SVM: {svm_acc:.4f}; {svm_f1:.4f}\n")
            f.write(f"LR:  {lr_acc:.4f}; {lr_f1:.4f}\n")
        with open("results.json", "w", encoding="utf-8") as f:
            json.dump({"SVM": {"accuracy": svm_acc, "f1": svm_f1}, "LR": {"accuracy": lr_acc, "f1": lr_f1}}, f, ensure_ascii=False, indent=2)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generic lab runner (SVM & LogReg)")
    p.add_argument("--csv", type=str, default=None, help="Path to CSV. If omitted, auto-pick in current dir.")
    p.add_argument("--target", type=str, default=DEFAULT_TARGET, help="Numeric target column name.")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Threshold for binarization (>= threshold -> 1).")
    p.add_argument("--test-size", type=float, default=0.25, help="Test split size (0..1).")
    p.add_argument("--random-state", type=int, default=42, help="Random seed.")
    p.add_argument("--id-cols", nargs="+", default=["user_id", "id", "index"], help="Identifier columns to drop.")
    p.add_argument("--save", action="store_true", help="Save results to results.txt and results.json")


def main(argv=None) -> int:
    args = parse_args(argv)
    csv_path = args.csv or find_csv_in_dir(".")
    print(f"📂 Using CSV: {csv_path}")

    df = read_csv_robust(csv_path)
    classification_training(
        data=df,
        target=args.target,
        threshold=args.threshold,
        id_cols=args.id_cols,
        test_size=args.test_size,
        random_state=args.random_state,
        save=args.save,
    )



if __name__ == "__main__":
    sys.exit(main())
