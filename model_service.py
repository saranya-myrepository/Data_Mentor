# ============================================================
# 📘 DataMentor: Model Training & Evaluation Service
# Supports both regression and classification workflows.
# ============================================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
import traceback

# ============================================================
# 🔹 Helper: Evaluate Regression
# ============================================================
def evaluate_regression(model, X_test, y_test):
    preds = model.predict(X_test)
    metrics = {
        "MAE": round(mean_absolute_error(y_test, preds), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, preds)), 4),
        "R2": round(r2_score(y_test, preds), 4)
    }
    return metrics

# ============================================================
# 🔹 Helper: Evaluate Classification
# ============================================================
def evaluate_classification(model, X_test, y_test):
    preds = model.predict(X_test)
    metrics = {
        "Accuracy": round(accuracy_score(y_test, preds), 4),
        "Precision": round(precision_score(y_test, preds, average='weighted', zero_division=0), 4),
        "Recall": round(recall_score(y_test, preds, average='weighted', zero_division=0), 4),
        "F1": round(f1_score(y_test, preds, average='weighted', zero_division=0), 4)
    }
    return metrics

# ============================================================
# 🔹 Main Function: Train and Evaluate Models
# ============================================================
def train_and_evaluate_models(preprocessor, X_train, X_test, y_train, y_test, problem_type):
    results = {}
    best_model_name = None
    best_score = -np.inf

    try:
        if "regression" in problem_type:
            models = {
                "LinearRegression": LinearRegression(),
                "RidgeRegression": Ridge(),
                "LassoRegression": Lasso(),
                "RandomForestRegressor": RandomForestRegressor(n_estimators=200, random_state=42),
                "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
                "XGBRegressor": XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
                "LGBMRegressor": LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
            }

            for name, model in models.items():
                model.fit(X_train, y_train)
                metrics = evaluate_regression(model, X_test, y_test)
                results[name] = metrics
                score = metrics["R2"]
                if score > best_score:
                    best_score = score
                    best_model_name = name

        else:  # Classification
            models = {
                "LogisticRegression": LogisticRegression(max_iter=1000),
                "RandomForestClassifier": RandomForestClassifier(n_estimators=200, random_state=42),
                "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
                "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=200),
                "LGBMClassifier": LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
            }

            for name, model in models.items():
                model.fit(X_train, y_train)
                metrics = evaluate_classification(model, X_test, y_test)
                results[name] = metrics
                score = metrics["F1"]
                if score > best_score:
                    best_score = score
                    best_model_name = name

        model_comparison = {
            "best_model_name": best_model_name,
            "total_models": len(models),
            "results": results
        }

        print(f"✅ Model training completed. Best model: {best_model_name}")
        return model_comparison, best_model_name

    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Model training failed: {str(e)}")
