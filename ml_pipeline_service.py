import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error

from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

def run_single_model(model, X_train, y_train, X_test, y_test, problem_type):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    if problem_type == "regression":
        return {
            "model": model.__class__.__name__,
            "score": r2_score(y_test, preds),
            "rmse": mean_squared_error(y_test, preds, squared=False)
        }
    else:
        return {
            "model": model.__class__.__name__,
            "score": accuracy_score(y_test, preds)
        }

def run_model_training(df, target, model_name, test_size):

    X = df.drop(columns=[target])
    y = df[target]

    problem_type = "regression" if y.dtype != "object" else "classification"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=42
    )

    # ---------- Compare All Models ----------
    if model_name.startswith("compare_all"):
        results = []
        if problem_type == "regression":
            models = [
                LinearRegression(), Lasso(),
                RandomForestRegressor(),
                SVR(), KNeighborsRegressor()
            ]
        else:
            models = [
                LogisticRegression(max_iter=500),
                RandomForestClassifier(),
                SVC(), KNeighborsClassifier()
            ]

        for m in models:
            try:
                results.append(run_single_model(
                    m, X_train, y_train, X_test, y_test, problem_type
                ))
            except:
                continue

        best = max(results, key=lambda x: x["score"])
        return {"best_model": best["model"], "results": results}

    # ---------- Single model ----------
    if model_name == "linear_regression":
        model = LinearRegression()
    elif model_name == "multiple_regression":
        model = LinearRegression()
    elif model_name == "lasso":
        model = Lasso()
    elif model_name == "random_forest_regressor":
        model = RandomForestRegressor()
    elif model_name == "svr":
        model = SVR()
    elif model_name == "knn_regressor":
        model = KNeighborsRegressor()
    elif model_name == "logistic_regression":
        model = LogisticRegression(max_iter=500)
    elif model_name == "random_forest_classifier":
        model = RandomForestClassifier()
    elif model_name == "svc":
        model = SVC()
    elif model_name == "knn_classifier":
        model = KNeighborsClassifier()
    else:
        model = LinearRegression()

    final = run_single_model(model, X_train, y_train, X_test, y_test, problem_type)
    final["model"] = model.__class__.__name__

    return final
