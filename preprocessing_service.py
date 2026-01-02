# ============================================================
# 📘 DataMentor: Preprocessing Service (SAFE VERSION)
# Fixes: "['Final_Score'] not found in axis"
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# ------------------------------------------------------------
# SAFE TARGET FINDER  (case-insensitive)
# ------------------------------------------------------------
def find_target_column(df, target_column):
    cols_lower = {c.lower(): c for c in df.columns}
    if target_column.lower() not in cols_lower:
        raise RuntimeError(
            f"Target column '{target_column}' missing after preprocessing.\n"
            f"Available columns: {list(df.columns)}"
        )
    return cols_lower[target_column.lower()]


# ============================================================
# 1️⃣ Missing Value Imputation
# ============================================================
def handle_missing_values(df: pd.DataFrame):
    df_clean = df.copy()

    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns

    # numeric
    if len(numeric_cols) > 0:
        imputer_num = SimpleImputer(strategy='median')
        df_clean[numeric_cols] = imputer_num.fit_transform(df_clean[numeric_cols])

    # categorical
    if len(categorical_cols) > 0:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        df_clean[categorical_cols] = imputer_cat.fit_transform(df_clean[categorical_cols])

    return df_clean


# ============================================================
# 2️⃣ Outlier Treatment (IQR Capping)
# ============================================================
def treat_outliers(df: pd.DataFrame):
    df_out = df.copy()
    for col in df_out.select_dtypes(include=[np.number]).columns:
        Q1 = df_out[col].quantile(0.25)
        Q3 = df_out[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_out[col] = np.clip(df_out[col], lower, upper)
    return df_out


# ============================================================
# 3️⃣ Encode Categorical Columns (BUT NEVER TARGET)
# ============================================================
def encode_features(df: pd.DataFrame, target_column):
    df_encoded = df.copy()
    encoders = {}

    tc = find_target_column(df, target_column)

    for col in df.columns:
        if col == tc:
            continue

        if df[col].dtype == "object" or str(df[col].dtype) == "category":
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le

    return df_encoded, encoders


# ============================================================
# 4️⃣ Feature Removal (Based on Profiling)
# ============================================================
def apply_feature_interpretation(df: pd.DataFrame, profiling_report: dict, target_column: str):
    df_clean = df.copy()
    feature_profiles = profiling_report.get("feature_profiles", {})

    tc = find_target_column(df, target_column)

    to_remove = []
    for col, meta in feature_profiles.items():
        if col == tc:
            continue  # never remove target
        if meta["interpretation"]["action"].lower() == "remove":
            to_remove.append(col)

    df_clean.drop(columns=to_remove, inplace=True, errors='ignore')
    return df_clean, to_remove


# ============================================================
# 5️⃣ Safe Split Train/Test
# ============================================================
def safe_split(df: pd.DataFrame, target_column: str, test_size=0.2, random_state=42):
    actual_tc = find_target_column(df, target_column)

    X = df.drop(columns=[actual_tc])
    y = df[actual_tc]

    strat = y if y.nunique() < 10 else None

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )


# ============================================================
# 6️⃣ Scaling
# ============================================================
def scale_features(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    return X_train_scaled, X_test_scaled, scaler


# ============================================================
# 7️⃣ MASTER PIPELINE (SAFE)
# ============================================================
def preprocess_data(df: pd.DataFrame, target_column: str, problem_type: str, profiling_report: dict):
    try:
        # 1. Profiling-based feature removal
        df_clean, removed_features = apply_feature_interpretation(df, profiling_report, target_column)

        # 2. Missing value imputation
        df_imputed = handle_missing_values(df_clean)

        # 3. Outlier treatment
        df_outlier_fixed = treat_outliers(df_imputed)

        # 4. Encoding (target protected)
        df_encoded, encoders = encode_features(df_outlier_fixed, target_column)

        # 5. Safe split
        X_train, X_test, y_train, y_test = safe_split(df_encoded, target_column)

        # 6. Scaling
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

        preprocessor = {
            "encoders": encoders,
            "scaler": scaler,
            "removed_features": removed_features
        }

        return preprocessor, X_train_scaled, X_test_scaled, y_train, y_test

    except Exception as e:
        raise RuntimeError(f"Preprocessing failed: {str(e)}")
