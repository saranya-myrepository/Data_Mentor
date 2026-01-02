import requests
import json

MODEL_NAME = "qwen2.5:3b"   # Faster and lighter model for your laptop

def qwen_chat(prompt: str):
    """
    Calls Qwen via Ollama HTTP API.
    Works cleanly on Windows (no TTY garbage, no escape codes).
    """
    try:
        response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False
            }
        )

        if response.status_code != 200:
            return f"Ollama API error: {response.text}"

        data = response.json()
        return data.get("response", "").strip()

    except Exception as e:
        return f"Error communicating with Ollama API: {str(e)}"
