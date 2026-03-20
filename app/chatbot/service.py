import os
from typing import Any, Dict, Tuple


DEFAULT_SUGGESTIONS = [
    "How do I predict a crop?",
    "Explain model insights",
    "Tell me about rice",
]


def _build_project_context(crop_info: Dict[str, Dict[str, str]], crop_library: Dict[str, Dict[str, str]]) -> str:
    crop_lines = []
    for crop_name in sorted(crop_info.keys()):
        info = crop_info.get(crop_name, {})
        lib = crop_library.get(crop_name, {})
        crop_lines.append(
            (
                f"- {crop_name}: desc={info.get('desc', 'NA')}; "
                f"soil={lib.get('soil', 'NA')}; ph={lib.get('ph', 'NA')}; "
                f"rainfall={lib.get('rain', 'NA')}; temp={lib.get('temp', 'NA')}"
            )
        )

    return (
        "Project: Precision Crop Recommender Flask app.\n"
        "Core features:\n"
        "- Predict Crop page: user enters N, P, K, temperature, humidity, pH, rainfall.\n"
        "- /predict endpoint returns top crop + top-5 confidence.\n"
        "- Crop Library page contains crop descriptions and ranges for some crops.\n"
        "- Model Insights page includes metrics (accuracy, precision, recall, F1, top-3, log loss).\n"
        "- This assistant should answer only project and agriculture context questions.\n"
        "Known crop entries:\n"
        + "\n".join(crop_lines)
    )


def _build_prompt_template(project_context: str, user_question: str) -> str:
    return f"""
You are the in-app Farm Assistant for this project.

Follow these rules strictly:
1. Answer based only on the project context below and user's question.
2. If exact data is unavailable, say clearly: "I don't have that exact value in this project data".
3. Keep answers practical, concise, and farmer-friendly.
4. Prefer plain text, max 6 sentences.
5. Do not invent API endpoints, metrics, or crop ranges.

PROJECT CONTEXT:
{project_context}

USER QUESTION:
{user_question}
""".strip()


def _load_gemini_client(api_key: str):
    try:
        import google.generativeai as genai  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Gemini SDK not installed. Install dependency: google-generativeai"
        ) from exc

    genai.configure(api_key=api_key)
    return genai


def _generate_with_gemini(prompt: str) -> str:
    api_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    model_name = (os.environ.get("GEMINI_MODEL") or "gemini-1.5-flash").strip()

    if not api_key:
        raise RuntimeError(
            "Gemini API key is missing. Set GEMINI_API_KEY in your .env file."
        )

    genai = _load_gemini_client(api_key)
    model = genai.GenerativeModel(model_name=model_name)
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.2,
            "max_output_tokens": 350,
        },
    )

    text = (getattr(response, "text", "") or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response.")
    return text


def generate_chatbot_reply(message: str, crop_info: Dict[str, Dict[str, str]], crop_library: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    user_message = (message or "").strip()
    if not user_message:
        return {
            "reply": "Please type a question so I can help.",
            "suggestions": DEFAULT_SUGGESTIONS,
        }

    project_context = _build_project_context(crop_info, crop_library)
    prompt = _build_prompt_template(project_context, user_message)

    try:
        answer = _generate_with_gemini(prompt)
        return {
            "reply": answer,
            "suggestions": DEFAULT_SUGGESTIONS,
        }
    except Exception as exc:
        return {
            "reply": (
                "I could not generate an AI response right now. "
                f"Reason: {exc}"
            ),
            "suggestions": [
                "Set GEMINI_API_KEY in .env",
                "How do I predict a crop?",
                "Open crop library",
            ],
        }
