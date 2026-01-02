# ============================================================
# 🤖 DataMentor Chatbot Service (with clean formatting)
# Uses Qwen 2.5 (3B) via Ollama HTTP API.
# ============================================================

from typing import Optional, Dict, Any
import re
import json

from app.services.llm_connector import qwen_chat

# -------------------------------------------------------------
# INTENT DETECTION
# -------------------------------------------------------------
def detect_intent(user_message: str) -> str:
    msg = (user_message or "").lower().strip()

    if len(msg) < 2 or re.fullmatch(r"[^a-zA-Z0-9]+", msg):
        return "unclear"

    if re.search(r'\b(datamentor|project|about|explain)\b', msg):
        return "project_info"
    if re.search(r'\b(miss|null|na|nan|empty|missing)\b', msg):
        return "missing_values"
    if re.search(r'\b(outlier|anomaly|extreme)\b', msg):
        return "outliers"
    if re.search(r'\b(correl|relationship|heatmap)\b', msg):
        return "correlation"
    if re.search(r'\b(feature|interpret)\b', msg):
        return "feature_interpretation"
    if re.search(r'\b(model|algorithm|train|predict)\b', msg):
        return "model_selection"
    if re.search(r'\b(metric|accuracy|recall|precision|rmse|auc)\b', msg):
        return "metrics"

    return "general"


# -------------------------------------------------------------
# SYSTEM PROMPT WITH CLEAN FORMATTING
# -------------------------------------------------------------
SYSTEM_PROMPT = """
You are DataMentor AI — the built-in assistant inside the **DataMentor Application**.

### 🧭 What DataMentor Is
DataMentor is a complete beginner-friendly data analysis platform with:

- 📥 Dataset Upload  
- 🔍 Automated Profiling  
- 🧩 Missing Value Detection & Imputation  
- 📊 Outlier Detection (IQR Method)  
- 🔤 Encoding (One-Hot, Label, Ordinal)  
- 📏 Scaling (Standard, MinMax)  
- 🔀 Train-Test Split  
- 🤖 ML Model Training & Comparison  
- 📈 Model Metrics  
- 🧠 Feature Interpretation  
- ⬇️ Downloading Cleaned Dataset  

### 🔒 RULES  
- ALWAYS describe the above when asked “What is DataMentor?”.  
- NEVER mention Alibaba Cloud or external tools.  
- ALWAYS respond in clean **markdown format**.  
- ALWAYS be beginner-friendly and step-by-step.  
- Use dataset summary when available.  
"""


# -------------------------------------------------------------
# PROMPT BUILDER
# -------------------------------------------------------------
def build_prompt(intent, user_message, profiling_data, model_results):
    dataset_block = f"\n### 📊 Dataset Summary\n```json\n{json.dumps(profiling_data, indent=2)}\n```" if profiling_data else ""
    model_block = f"\n### 🤖 Model Results\n```json\n{json.dumps(model_results, indent=2)}\n```" if model_results else ""

    return (
        SYSTEM_PROMPT
        + dataset_block
        + model_block
        + f"\n### ❓ User Question\n{user_message}\n"
        + f"\n### 🎯 Detected Intent: **{intent}**"
        + "\n\n### 🧠 Give a clean, structured, bullet-point explanation:\n"
    )


# -------------------------------------------------------------
# MAIN ENTRY POINT
# -------------------------------------------------------------
def get_chatbot_reply(user_message, profiling_data=None, model_results=None):
    try:
        intent = detect_intent(user_message)
        prompt = build_prompt(intent, user_message, profiling_data, model_results)
        response = qwen_chat(prompt)
        return response

    except Exception as e:
        return f"⚠️ Error: {str(e)}"
