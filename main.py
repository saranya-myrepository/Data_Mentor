# ============================================================
# 🚀 DataMentor: Intelligent Dataset Understanding & ML Pipeline
# (CLEAN VERSION — All helpers, SHAP, Plotly, AutoML)
# ============================================================
import os
import uuid
import datetime
import io
import json
import base64
from zipfile import ZipFile
from werkzeug.utils import secure_filename

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, jsonify, send_file
)
import pandas as pd
import numpy as np

# ============================================================
# 🔵 PLOTLY + SHAP + ML IMPORTS
# ============================================================
import shap
import plotly.graph_objects as go
import plotly.express as px
from app.reports.final_report_generator import create_final_report
from app.shared_state import DATA_CACHE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc,
    r2_score, mean_squared_error, mean_absolute_error
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
import joblib
# ============================================================
# 🔵 ML MODEL REGISTRY — USED FOR AUTOML + MANUAL SELECTION
# ============================================================
CLASSIFICATION_MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=300),
    "Random Forest Classifier": RandomForestClassifier(),
    "Decision Tree Classifier": DecisionTreeClassifier(),
    "SVM Classifier": SVC(probability=True),
    "KNN Classifier": KNeighborsClassifier(),
    "Gradient Boosting Classifier": GradientBoostingClassifier()
}

REGRESSION_MODELS = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(),
    "Decision Tree Regressor": DecisionTreeRegressor(),
    "KNN Regressor": KNeighborsRegressor(),
    "Gradient Boosting Regressor": GradientBoostingRegressor()
}

# ============================================================
# 🎨 PLOTLY BLUE THEME (MATCHES ALL YOUR UI PAGES)
# ============================================================
def apply_blue_theme(fig):
    fig.update_layout(
        template="plotly_white",
        title_font=dict(size=22, color="#0d47a1"),
        font=dict(size=14, color="#0d47a1"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#e3f2fd"),
        yaxis=dict(showgrid=True, gridcolor="#e3f2fd")
    )
    return fig
# ============================================================
# 📊 METRIC CALCULATORS
# ============================================================
def compute_classification_metrics(y_test, preds):
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, average="weighted", zero_division=0),
        "recall": recall_score(y_test, preds, average="weighted", zero_division=0),
        "f1": f1_score(y_test, preds, average="weighted", zero_division=0)
    }


def compute_regression_metrics(y_test, preds):
    return {
        "r2": r2_score(y_test, preds),
        "rmse": mean_squared_error(y_test, preds) ** 0.5,
        "mae": mean_absolute_error(y_test, preds)
    }
# ============================================================
# 📈 PLOT GENERATORS (Confusion Matrix, ROC, Residuals, FI)
# ============================================================

def plot_confusion_matrix(cm):
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues")
    fig.update_layout(title="Confusion Matrix")
    return apply_blue_theme(fig)

def plot_roc(y_test, probas):
    fpr, tpr, _ = roc_curve(y_test, probas)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {roc_auc:.3f}"))
    fig.update_layout(title="ROC Curve")
    return apply_blue_theme(fig)

def plot_residuals(y_test, preds):
    residuals = y_test - preds
    fig = px.scatter(x=preds, y=residuals,
                     labels={"x": "Predicted", "y": "Residuals"})
    fig.update_layout(title="Residual Plot")
    return apply_blue_theme(fig)

def plot_feature_importance(model, features):
    if not hasattr(model, "feature_importances_"):
        return None
    fi = model.feature_importances_
    fig = px.bar(x=fi, y=features, orientation='h')
    fig.update_layout(title="Feature Importance")
    return apply_blue_theme(fig)

# ============================================================
# 🧠 SHAP SUMMARY (TREE + KERNEL FALLBACK)
# ============================================================

def generate_shap_summary(model, X_sample):
    """
    Returns: shap_html, shap_values
    """
    try:
        if "Forest" in str(type(model)) or "Tree" in str(type(model)):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(model.predict, X_sample)


        shap_values = explainer.shap_values(X_sample)


        shap_html = """
            <script src="https://cdn.jsdelivr.net/npm/shap@latest/dist/shap.min.js"></script>
        """
        return shap_html, shap_values
    except Exception as e:
        print("SHAP ERROR:", e)
        return None, None


# ============================================================
# 🤖 AUTOML ENGINE — TRAIN ALL MODELS + RANK
# ============================================================


def run_automl(task, X_train, X_test, y_train, y_test):
    results = []
    model_dict = CLASSIFICATION_MODELS if task == "classification" else REGRESSION_MODELS


    for model_name, model in model_dict.items():
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)


            if task == "classification":
                metrics = compute_classification_metrics(y_test, preds)
                score = metrics["accuracy"]
            else:
                metrics = compute_regression_metrics(y_test, preds)
                score = metrics["r2"]


            results.append({
                "name": model_name,
                "model": model,
                "metrics": metrics,
                "score": score
            })


        except Exception as e:
            print(f"Error training {model_name}: {e}")


    return sorted(results, key=lambda x: x["score"], reverse=True)


# ============================================================
# 🔹 Your existing app imports (already loaded in Part 1)
# ============================================================


from app.config import Config, create_required_directories
from app.utils.data_loader import save_and_load_data
from app.services.profiling_service import profile_data
from app.services.chatbot_service import get_chatbot_reply
from app.services.preprocessing_service import preprocess_data
from app.services.model_service import train_and_evaluate_models
from app.reports.report_generator import (
    generate_visualizations,
    generate_final_report_summary,
    export_pdf_report
)


from app.utils.missingness import (
    detect_missing_types,
    apply_missing_value_imputation,
    detect_outlier_types,
    apply_outlier_strategy
)


# ============================================================
# 🔹 App Setup
# ============================================================


app = Flask(__name__, template_folder='templates', static_folder='static')
app.config.from_object(Config)
app.secret_key = Config.SECRET_KEY
app.config['PERMANENT_SESSION_LIFETIME'] = datetime.timedelta(hours=1)


create_required_directories()


ALLOWED_EXT = {".csv", ".xls", ".xlsx"}


def allowed_filename(filename):
    if not filename:
        return False
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXT




# ============================================================
# 1️⃣ Landing Page
# ============================================================


@app.route("/")
def index():
    return render_template("landing.html")




# ============================================================
# 2️⃣ Upload Form UI
# ============================================================


@app.route("/upload_form")
def upload_form():
    return render_template(
        "index.html",
        error=session.pop("error", None),
        success_message=session.pop("success_message", None)
    )




# ============================================================
# 3️⃣ Column Fetch (for preview)
# ============================================================


@app.route("/get_columns", methods=["POST"])
def get_columns():
    file = request.files.get("dataset")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400


    filename = secure_filename(file.filename)
    if not allowed_filename(filename):
        return jsonify({"error": "Unsupported file type"}), 400


    temp_name = f"temp_{uuid.uuid4().hex}_{filename}"
    temp_path = os.path.join(Config.UPLOAD_FOLDER, temp_name)
    file.save(temp_path)


    df = (
        pd.read_excel(temp_path)
        if filename.lower().endswith((".xls", ".xlsx"))
        else pd.read_csv(temp_path)
    )


    session["uploaded_file_path"] = temp_path


    return jsonify({
        "columns": list(df.columns),
        "preview_html": df.head().to_html(classes="preview-table", index=False)
    })




# ============================================================
# 4️⃣ Upload Final Submit → Profiling
# ============================================================


@app.route("/upload", methods=["POST"])
def upload_data():
    try:
        file = request.files.get("dataset")
        temp_path = None


        if file and file.filename:
            df, error = save_and_load_data(file)
            if error:
                session["error"] = error
                return redirect(url_for("upload_form"))
        else:
            # use previously stored temp file
            temp_path = session.get("uploaded_file_path")
            if not temp_path or not os.path.exists(temp_path):
                session["error"] = "Upload a dataset first."
                return redirect(url_for("upload_form"))


            df = (
                pd.read_excel(temp_path)
                if temp_path.endswith((".xls", ".xlsx"))
                else pd.read_csv(temp_path)
            )


        target_column = request.form.get("target_column")
        problem_type = request.form.get("problem_type")
        business_objective = request.form.get("business_objective", "")


        if not target_column or target_column not in df.columns:
            session["error"] = "Select a valid target column."
            return redirect(url_for("upload_form"))


        # Profiling
        profiling_report = profile_data(df, target_column)
        profiling_report["missing_type_info"] = detect_missing_types(df)
        profiling_report["outlier_type_info"] = detect_outlier_types(df, target_column)


        # Cache session
        session_id = str(uuid.uuid4())
        session["session_id"] = session_id


        DATA_CACHE[session_id] = {
            "raw_df": df,
            "profiling_data": profiling_report,
            "user_inputs": {
                "target_column": target_column,
                "problem_type": problem_type,
                "business_objective": business_objective
            }
        }


        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        session.pop("uploaded_file_path", None)


        return redirect(url_for("confirm_profile", session_id=session_id))


    except Exception as e:
        session["error"] = f"Upload failed: {str(e)}"
        return redirect(url_for("upload_form"))




# ============================================================
# 5️⃣ Profiling Dashboard
# ============================================================


@app.route("/confirm_profile/<session_id>")
def confirm_profile(session_id):
    sd = DATA_CACHE.get(session_id)
    if not sd:
        session["error"] = "Session expired, upload again."
        return redirect(url_for("upload_form"))


    df = sd["raw_df"]
    target_col = sd["user_inputs"]["target_column"]


    from app.reports.visualization_utils import (
        plot_correlation_heatmap,
        plot_feature_relation
    )


    correlation_html = plot_correlation_heatmap(df, target_col)
    feature_plots = {
        col: plot_feature_relation(df, col, target_col) for col in df.columns
    }


    return render_template(
        "confirm_profile.html",
        report=sd["profiling_data"],
        user_inputs=sd["user_inputs"],
        correlation_html=correlation_html,
        feature_plots=feature_plots,
        session_id=session_id
    )




# ============================================================
# 6️⃣ Feature Interpretation Page
# ============================================================


@app.route("/feature_interpretation/<session_id>")
def feature_interpretation(session_id):
    sd = DATA_CACHE.get(session_id)
    if not sd:
        return redirect(url_for("upload_form"))


    return render_template(
        "feature_interpretation.html",
        report=sd["profiling_data"],
        user_inputs=sd["user_inputs"],
        session_id=session_id
    )




# ============================================================
# 7️⃣ Preprocess + Modeling Pipeline
# ============================================================


@app.route("/start_pipeline/<session_id>", methods=["POST"])
def start_pipeline(session_id):
    sd = DATA_CACHE.get(session_id)
    if not sd:
        return redirect(url_for("upload_form"))


    try:
        df = sd["raw_df"]
        target = sd["user_inputs"]["target_column"]
        problem_type = sd["user_inputs"]["problem_type"]
        profile = sd["profiling_data"]


        preprocessor, X_train, X_test, y_train, y_test = preprocess_data(
            df.copy(), target, problem_type, profile
        )


        clean_path = os.path.join(Config.UPLOAD_FOLDER, f"{session_id}_processed.csv")
        pd.concat([X_train, y_train], axis=1).to_csv(clean_path, index=False)
        sd["processed_file"] = clean_path
        sd["profiling_data"]["cleaned_available"] = True


        # Model training
        model_results, best_model_name = train_and_evaluate_models(
            preprocessor, X_train, X_test, y_train, y_test, problem_type
        )


        visuals = generate_visualizations(df, profile)
        summary = generate_final_report_summary(profile, model_results, best_model_name)


        pdf_path = export_pdf_report(session_id, profile, model_results, Config.UPLOAD_FOLDER)


        sd["final_results"] = {
            "report_summary": summary,
            "model_results": model_results,
            "visualizations": visuals,
            "pdf_path": pdf_path
        }


        return redirect(url_for("results", session_id=session_id))


    except Exception as e:
        session["error"] = f"Pipeline error: {str(e)}"
        return redirect(url_for("confirm_profile", session_id=session_id))




# ============================================================
# 8️⃣ Results Page
# ============================================================


@app.route("/results/<session_id>")
def results(session_id):
    sd = DATA_CACHE.get(session_id)
    if not sd or "final_results" not in sd:
        return redirect(url_for("upload_form"))


    return render_template(
        "results.html",
        report=sd["final_results"]["report_summary"],
        visualizations=sd["final_results"]["visualizations"],
        session_id=session_id
    )
# ============================================================
# 9️⃣ Downloads (Existing System)
# ============================================================


@app.route("/download_processed/<session_id>")
def download_processed(session_id):
    path = os.path.join(Config.UPLOAD_FOLDER, f"{session_id}_processed.csv")
    return send_file(path, as_attachment=True)




@app.route("/download_correlation_map/<session_id>")
def download_correlation_map(session_id):
    try:
        sd = DATA_CACHE.get(session_id)
        if not sd:
            session["error"] = "Session expired."
            return redirect(url_for('upload_form'))


        df = sd["raw_df"]
        target_col = sd["user_inputs"]["target_column"]


        os.makedirs(Config.REPORTS_FOLDER, exist_ok=True)


        try:
            from app.reports.visualization_utils import plot_correlation_heatmap
            html = plot_correlation_heatmap(df, target_col)


            out_path = os.path.join(Config.REPORTS_FOLDER, f"{session_id}_corrmap.html")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(html)
            return send_file(out_path, as_attachment=True)


        except Exception:
            corr = df.select_dtypes(include=['number']).corr()
            out_path = os.path.join(Config.REPORTS_FOLDER, f"{session_id}_corrmap.csv")
            corr.to_csv(out_path)
            return send_file(out_path, as_attachment=True)


    except Exception as e:
        session["error"] = f"Failed: {str(e)}"
        return redirect(url_for('confirm_profile', session_id=session_id))




@app.route("/download_report/<session_id>")
def download_report(session_id):
    pdf_path = DATA_CACHE[session_id]["final_results"]["pdf_path"]
    return send_file(pdf_path, as_attachment=True)




# ============================================================
# 🔟 Chatbot API
# ============================================================


@app.route("/chatbot_response", methods=["POST"])
def chatbot_response():
    msg = request.json.get("message", "")
    sid = session.get("session_id")


    profile = DATA_CACHE.get(sid, {}).get("profiling_data")
    model_data = DATA_CACHE.get(sid, {}).get("final_results", {}).get("model_results")


    reply = get_chatbot_reply(msg, profile, model_data)
    return jsonify({"reply": reply})




# ============================================================
# 1️⃣1️⃣ Preprocess Only (Download Cleaned CSV)
# ============================================================


@app.route("/preprocess_only/<session_id>", methods=["POST"])
def preprocess_only(session_id):
    sd = DATA_CACHE.get(session_id)
    df = sd["raw_df"]
    target = sd["user_inputs"]["target_column"]
    task = sd["user_inputs"]["problem_type"]
    profile = sd["profiling_data"]


    preprocessor, X_train, X_test, y_train, y_test = preprocess_data(
        df.copy(), target, task, profile
    )


    path = os.path.join(Config.UPLOAD_FOLDER, f"{session_id}_processed.csv")
    pd.concat([X_train, y_train], axis=1).to_csv(path, index=False)


    return send_file(path, as_attachment=True)




# ============================================================
# 1️⃣2️⃣ Missing Value Imputation Download
# ============================================================


@app.route("/apply_imputation/<session_id>", methods=["POST"])
def apply_imputation(session_id):
    df = DATA_CACHE[session_id]["raw_df"]
    method = request.form.get("method", "mean")


    cleaned = apply_missing_value_imputation(df.copy(), method)
    out = os.path.join(Config.UPLOAD_FOLDER, f"{session_id}_imputed.csv")
    cleaned.to_csv(out, index=False)


    return send_file(out, as_attachment=True)




# ============================================================
# 1️⃣3️⃣ Outlier Handling Download
# ============================================================


@app.route("/apply_outlier_handling/<session_id>", methods=["POST"])
def apply_outlier_handling(session_id):
    df = DATA_CACHE[session_id]["raw_df"]
    strategy = request.form.get("strategy", "cap")


    cleaned = apply_outlier_strategy(df.copy(), strategy)
    out = os.path.join(Config.UPLOAD_FOLDER, f"{session_id}_outliers_cleaned.csv")
    cleaned.to_csv(out, index=False)


    return send_file(out, as_attachment=True)


# ============================================================
# 🚀 NEW 4-STEP ML PIPELINE (CLEAN + CORRECTED)
# ============================================================


print("🔵 ML PIPELINE ROUTES LOADED")


# ------------------------------------------------------------
# STEP 1 — Upload Preprocessed CSV
# ------------------------------------------------------------
@app.route("/ml_pipeline_upload", methods=["GET", "POST"])
def ml_pipeline_upload():


    if request.method == "POST":
        file = request.files.get("dataset")


        if not file:
            return render_template("ml_pipeline_upload.html",
                                   error="Please upload a CSV file.")


        try:
            df = pd.read_csv(file)
        except Exception as e:
            return render_template(
                "ml_pipeline_upload.html",
                error=f"Unable to read the CSV file. Error: {str(e)}"
            )


        df.columns = df.columns.str.strip()


        # Store DF in new ml_id session
        ml_id = str(uuid.uuid4())
        session["ml_id"] = ml_id
        DATA_CACHE[ml_id] = {"df": df}


        return redirect(url_for("ml_pipeline_target"))


    return render_template("ml_pipeline_upload.html")




# ------------------------------------------------------------
# STEP 2 — Select Target Variable
# ------------------------------------------------------------
@app.route("/ml_pipeline_target", methods=["GET", "POST"])
def ml_pipeline_target():
    ml_id = session.get("ml_id")


    # Validation
    if not ml_id or ml_id not in DATA_CACHE:
        return redirect(url_for("ml_pipeline_upload"))


    df = DATA_CACHE[ml_id]["df"]


    if request.method == "POST":
        target = request.form.get("target_column")


        if not target or target not in df.columns:
            return render_template(
                "ml_pipeline_target.html",
                columns=df.columns,
                error="Please select a valid target variable."
            )


        session["ml_target"] = target
        return redirect(url_for("ml_pipeline_model"))


    return render_template("ml_pipeline_target.html", columns=df.columns)


####Step 3
# ------------------------------------------------------------
# STEP 3 — AutoML + Multi-Model Selection (Top-3 Recommended)
# ------------------------------------------------------------
@app.route("/ml_pipeline_model", methods=["GET", "POST"])
def ml_pipeline_model():


    ml_id = session.get("ml_id")
    target = session.get("ml_target")


    # Validate session
    if not ml_id or ml_id not in DATA_CACHE:
        return redirect(url_for("ml_pipeline_upload"))
    if not target:
        return redirect(url_for("ml_pipeline_target"))


    df = DATA_CACHE[ml_id]["df"]


    # =========================
    # Prepare X, y
    # =========================
    y = df[target]
    X = df.drop(columns=[target])


    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(X.mean())


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    # =========================
    # Detect task
    # =========================
    task = "classification" if (y.dtype == "object" or y.nunique() <= 10) else "regression"
    session["ml_task"] = task


    # ------------------------------------------------------------
    # POST: user selected models manually (max 3)
    # ------------------------------------------------------------
    if request.method == "POST":
        selected = request.form.getlist("models_selected")


        if not selected:
            return render_template(
                "ml_pipeline_model.html",
                task=task,
                error="Please select at least one model.",
                leaderboard=session.get("ml_leaderboard"),
                model_groups=session.get("ml_model_groups")
            )


        if len(selected) > 3:
            return render_template(
                "ml_pipeline_model.html",
                task=task,
                error="You can select a maximum of 3 models.",
                leaderboard=session.get("ml_leaderboard"),
                model_groups=session.get("ml_model_groups")
            )


        session["ml_selected_models"] = selected
        return redirect(url_for("ml_pipeline_results"))


    # ------------------------------------------------------------
    # GET: AutoML Evaluation
    # ------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )


    model_dict = CLASSIFICATION_MODELS if task == "classification" else REGRESSION_MODELS


    # Run AutoML (evaluate all)
    leaderboard_full = run_automl(task, X_train, X_test, y_train, y_test)


    # Save leaderboard for template
    leaderboard_clean = [
        {"name": m["name"], "score": round(m["score"], 4)}
        for m in leaderboard_full
    ]
    session["ml_leaderboard"] = leaderboard_clean


    # Top-3
    top3 = leaderboard_full[:3]
    recommended_list = [m["name"] for m in top3]


    # Save recommended
    session["ml_recommended"] = recommended_list


    # Group models for UI
    model_groups = {
        "Top-3 Recommended": recommended_list,
        "Basic Models": [
            m for m in model_dict.keys()
            if m not in recommended_list
        ],
    }
    session["ml_model_groups"] = model_groups


    # ------------------------------------------------------------
    # Render Step-3 UI
    # ------------------------------------------------------------
    return render_template(
        "ml_pipeline_model.html",
        task=task,
        leaderboard=leaderboard_clean,
        model_groups=model_groups
    )


# ------------------------------------------------------------
# STEP 4 — Final Model Training + Multi-Model Evaluation
# ------------------------------------------------------------
@app.route("/ml_pipeline_results")
def ml_pipeline_results():


    ml_id = session.get("ml_id")
    target = session.get("ml_target")
    task = session.get("ml_task")
    selected_models = session.get("ml_selected_models")


    # Validate session
    if not ml_id or ml_id not in DATA_CACHE:
        return redirect(url_for("ml_pipeline_upload"))


    if not target:
        return redirect(url_for("ml_pipeline_target"))


    if not selected_models or len(selected_models) == 0:
        return redirect(url_for("ml_pipeline_model"))


    # Only allow max 3 models
    selected_models = selected_models[:3]


    df = DATA_CACHE[ml_id]["df"]


    # ------------------------------------------------------------
    # PREPARE DATA
    # ------------------------------------------------------------
    y = df[target]
    X = df.drop(columns=[target])


    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(X.mean())
    feature_names = list(X.columns)


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )


    # ------------------------------------------------------------
    # MODEL REGISTRY
    # ------------------------------------------------------------
    model_registry = CLASSIFICATION_MODELS if task == "classification" else REGRESSION_MODELS


    # ------------------------------------------------------------
    # STORAGE FOR RESULTS
    # ------------------------------------------------------------
    results = []               # list of dicts: name, model obj, metrics
    radar_labels = []          # metric names
    radar_values = []          # values for radar chart
    learning_curves = {}       # learning curves per model
    metric_table = {}          # comparison table (🔥 important fix)


    # ------------------------------------------------------------
    # TRAIN/EVALUATE ALL SELECTED MODELS
    # ------------------------------------------------------------
    import numpy as np


    for model_name in selected_models:


        if model_name not in model_registry:
            continue


        model = model_registry[model_name]


        # ---- TRAIN MODEL ----
        model.fit(X_train, y_train)
        preds = model.predict(X_test)


        # ---- METRICS ----
        if task == "classification":
            metrics = {
                "accuracy": accuracy_score(y_test, preds),
                "precision": precision_score(y_test, preds, average="weighted", zero_division=0),
                "recall": recall_score(y_test, preds, average="weighted", zero_division=0),
                "f1": f1_score(y_test, preds, average="weighted", zero_division=0),
            }
        else:
            metrics = {
                "r2": r2_score(y_test, preds),
                "rmse": mean_squared_error(y_test, preds) ** 0.5,
                "mae": mean_absolute_error(y_test, preds),
            }


        # Save metric table row
        metric_table[model_name] = metrics


        # Save radar values
        radar_labels = list(metrics.keys())
        radar_values.append({
            "model": model_name,
            "values": list(metrics.values())
        })


        # ---- SYNTHETIC LEARNING CURVES ----
        sizes = np.linspace(0.1, 1.0, 8)
        train_scores = []
        test_scores = []


        for s in sizes:
            n = int(len(X_train) * s)
            X_sub = X_train[:n]
            y_sub = y_train[:n]


            m = model_registry[model_name]
            m.fit(X_sub, y_sub)


            train_pred = m.predict(X_sub)
            test_pred = m.predict(X_test)


            if task == "classification":
                train_scores.append(accuracy_score(y_sub, train_pred))
                test_scores.append(accuracy_score(y_test, test_pred))
            else:
                train_scores.append(r2_score(y_sub, train_pred))
                test_scores.append(r2_score(y_test, test_pred))


        learning_curves[model_name] = {
            "sizes": list(sizes),
            "train": train_scores,
            "test": test_scores
        }


        # Save results
        results.append({
            "name": model_name,
            "model": model,
            "metrics": metrics
        })


    # ------------------------------------------------------------
    # CHOOSE BEST MODEL
    # ------------------------------------------------------------
    if task == "classification":
        best_model = max(results, key=lambda x: x["metrics"]["accuracy"])
    else:
        best_model = max(results, key=lambda x: x["metrics"]["r2"])


    best_model_name = best_model["name"]
    best_model_obj = best_model["model"]


    # ------------------------------------------------------------
    # CONFUSION MATRIX (classification)
    # ------------------------------------------------------------
    confusion_html = None
    if task == "classification":
        best_preds = best_model_obj.predict(X_test)
        cm = confusion_matrix(y_test, best_preds)


        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues")
        fig.update_layout(title=f"Confusion Matrix — {best_model_name}")
        fig = apply_blue_theme(fig)
        confusion_html = fig.to_html(full_html=False)


    # ------------------------------------------------------------
    # RESIDUAL PLOT (regression)
    # ------------------------------------------------------------
    residual_html = None
    if task == "regression":
        best_preds = best_model_obj.predict(X_test)
        residuals = y_test - best_preds


        fig = px.scatter(
            x=best_preds,
            y=residuals,
            labels={"x": "Predicted", "y": "Residuals"}
        )
        fig.update_layout(title=f"Residual Plot — {best_model_name}")
        fig = apply_blue_theme(fig)
        residual_html = fig.to_html(full_html=False)


    # ------------------------------------------------------------
    # FEATURE IMPORTANCE (tree models only)
    # ------------------------------------------------------------
    feature_importance_html = None
    if hasattr(best_model_obj, "feature_importances_"):
        fig = px.bar(
            x=best_model_obj.feature_importances_,
            y=feature_names,
            orientation="h",
            title=f"Feature Importance — {best_model_name}"
        )
        fig = apply_blue_theme(fig)
        feature_importance_html = fig.to_html(full_html=False)


    # ------------------------------------------------------------
    # SHAP (tree models only)
    # ------------------------------------------------------------
    shap_html = None
    try:
        shap_html, _ = generate_shap_summary(best_model_obj, X_test[:50])
    except:
        shap_html = None


        # ============================================================
    # 📌 STEP 3: SAVE CHARTS AND IMPORTANT VALUES FOR PDF REPORT
    # ============================================================


    # Folder for saving chart images
    reports_dir = os.path.join("static", "pipeline_reports")
    os.makedirs(reports_dir, exist_ok=True)


    chart_paths = []


    # ---------------------------
    # Save Confusion Matrix Image
    # ---------------------------
    if task == "classification":
        try:
            fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues")
            fig.update_layout(title=f"Confusion Matrix — {best_model_name}")
            img_path = f"{reports_dir}/confusion_{ml_id}.png"
            fig.write_image(img_path)
            chart_paths.append(img_path)
        except:
            pass


    # -----------------------
    # Save Residual Plot
    # -----------------------
    if task == "regression":
        try:
            fig = px.scatter(
                x=best_preds,
                y=residuals,
                labels={"x": "Predicted", "y": "Residuals"}
            )
            fig.update_layout(title=f"Residual Plot — {best_model_name}")
            img_path = f"{reports_dir}/residual_{ml_id}.png"
            fig.write_image(img_path)
            chart_paths.append(img_path)
        except:
            pass


    # -----------------------------
    # Save Feature Importance Image
    # -----------------------------
    if hasattr(best_model_obj, "feature_importances_"):
        try:
            fig = px.bar(
                x=best_model_obj.feature_importances_,
                y=feature_names,
                orientation="h",
                title=f"Feature Importance — {best_model_name}"
            )
            img_path = f"{reports_dir}/feature_importance_{ml_id}.png"
            fig.write_image(img_path)
            chart_paths.append(img_path)
        except:
            pass


    # -----------------------------
    # Store all values for PDF use
    # -----------------------------
    session["metric_table_cache"] = metric_table
    session["best_model_name_cache"] = best_model_name


    # objective will be auto-generated later
    session["objective_text_cache"] = None


    # Path list of all saved charts
    session["chart_paths_cache"] = chart_paths






#Added for report


    # ------------------------------------------------------------
    # RETURN ALL VALUES TO TEMPLATE
    # ------------------------------------------------------------
    return render_template(
        "ml_pipeline_results.html",
        best_model_name=best_model_name,
        metric_table=metric_table,
        radar_labels=radar_labels,
        radar_values=radar_values,
        learning_curves=learning_curves,     # optional: if template uses it
        task=task,
        confusion_html=confusion_html,
        residual_html=residual_html,
        feature_importance_html=feature_importance_html,
        shap_html=shap_html
    )


###For report page routing ####
from app.reports.final_report_generator import create_final_report


@app.route("/generate_report", methods=["POST"])
def generate_report():
    ml_id = session.get("ml_id")
    pdf_path = create_final_report(ml_id)
    return send_file(pdf_path, as_attachment=True)

#=============================================================
# 🚀 RUN APP
# ============================================================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

