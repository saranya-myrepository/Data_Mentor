# ============================================================
# 📘 DataMentor: Profiling Service
# Performs automatic dataset analysis for missing values,
# outliers, correlations, and feature interpretation.
# ============================================================

import pandas as pd
import numpy as np
from scipy import stats

# ============================================================
# 🔹 1️⃣  Missing Value Analysis
# ============================================================
def analyze_missing_values(df: pd.DataFrame):
    missing_summary = {}
    total_missing_cells = int(df.isnull().sum().sum())

    for col in df.columns:
        missing_pct = (df[col].isna().sum() / len(df)) * 100
        missing_summary[col] = {
            "missing_pct": round(missing_pct, 2),
            "missing_count": int(df[col].isna().sum())
        }

    return {
        "total_missing_cells": total_missing_cells,
        "per_feature": missing_summary
    }

# ============================================================
# 🔹 2️⃣  Outlier Detection (IQR Method)
# ============================================================
def detect_outliers(df: pd.DataFrame):
    outlier_summary = {}

    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outlier_pct = (outliers / len(df)) * 100

        outlier_summary[col] = {
            "outlier_count": int(outliers),
            "outlier_pct": round(outlier_pct, 2)
        }

    return outlier_summary

# ============================================================
# 🔹 3️⃣  Feature Interpretation Logic
# ============================================================
def interpret_features(df: pd.DataFrame, missing_data, outlier_data):
    feature_profiles = {}

    for col in df.columns:
        dtype = str(df[col].dtype)
        unique_count = df[col].nunique()
        missing_pct = missing_data["per_feature"].get(col, {}).get("missing_pct", 0)
        outlier_pct = outlier_data.get(col, {}).get("outlier_pct", 0)

        interpretation = {"utility": "High", "action": "Keep", "reason": []}

        # --- Missing values impact ---
        if missing_pct > 60:
            interpretation.update({
                "utility": "Low", "action": "Remove",
                "reason": ["More than 60% values missing."]
            })
        elif 20 < missing_pct <= 60:
            interpretation.update({
                "utility": "Low", "action": "Review",
                "reason": ["Moderate missing values (20%-60%)."]
            })
        elif 0 < missing_pct <= 20:
            interpretation["reason"].append("Few missing values (<20%), safe to impute.")

        # --- Outlier impact ---
        if outlier_pct > 10:
            interpretation.update({
                "utility": "Low", "action": "Review",
                "reason": interpretation["reason"] + ["High outlier percentage (>10%)."]
            })

        # --- Unique count analysis ---
        if unique_count == 1:
            interpretation.update({
                "utility": "None", "action": "Remove",
                "reason": ["Constant feature (only one unique value)."]
            })
        elif unique_count == len(df):
            interpretation.update({
                "utility": "None", "action": "Remove",
                "reason": ["Likely an ID column (unique for every row)."]
            })
        elif unique_count > 50 and df[col].dtype == "object":
            interpretation.update({
                "utility": "Low", "action": "Review",
                "reason": interpretation["reason"] + ["High-cardinality categorical feature."]
            })

        # --- Default stats (mean/mode) ---
        stats_dict = {}
        if np.issubdtype(df[col].dtype, np.number):
            stats_dict["mean"] = round(df[col].mean(skipna=True), 3)
        else:
            stats_dict["mode"] = df[col].mode().iloc[0] if not df[col].mode().empty else None

        feature_profiles[col] = {
            "dtype": dtype,
            "unique_count": int(unique_count),
            "missing_pct": round(missing_pct, 2),
            "outlier_pct": round(outlier_pct, 2),
            "interpretation": interpretation,
            "stats": stats_dict
        }

    return feature_profiles

# ============================================================
# 🔹 4️⃣  Correlation Analysis
# ============================================================
def compute_correlations(df: pd.DataFrame, target_column: str):
    corr_summary = {}
    if target_column not in df.columns:
        return {}

    try:
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        if target_column in corr_matrix.columns:
            target_corr = corr_matrix[target_column].dropna().sort_values(key=abs, ascending=False)
            corr_summary = target_corr.head(10).to_dict()
    except Exception:
        pass

    return corr_summary

# ============================================================
# 🔹 5️⃣  Generate Full Profiling Report
# ============================================================
def profile_data(df: pd.DataFrame, target_column: str):
    try:
        # --- Drop rows where target is missing ---
        df = df.dropna(subset=[target_column])

        # --- Basic dataset summary ---
        general_stats = {
            "total_rows_after_target_check": len(df),
            "total_columns": df.shape[1],
            "rows_removed_target_missing": int(df[target_column].isna().sum())
        }

        # --- Compute insights ---
        missing_data = analyze_missing_values(df)
        outlier_data = detect_outliers(df)
        feature_profiles = interpret_features(df, missing_data, outlier_data)
        correlation_summary = compute_correlations(df, target_column)

        profiling_report = {
            "general_stats": general_stats,
            "missing_data_summary": missing_data,
            "outlier_summary": outlier_data,
            "feature_profiles": feature_profiles,
            "correlation_summary": correlation_summary
        }

        return profiling_report

    except Exception as e:
        raise RuntimeError(f"Profiling failed: {str(e)}")
