"""
DermIQ — Gemini API Integration
Product recommendation formatting for Pakistani users.
"""

import json
import os
import re

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_MODELS = ["gemini-1.5-flash", "gemini-2.5-flash", "gemini-2.0-flash"]

PRODUCT_RECOMMENDATION_PROMPT = """You are DermIQ, a friendly and professional AI skincare advisor for Pakistani users.

The user completed a skin profile questionnaire. Based on their answers, these {count} Hemani Herbals products are the best matches:

{product_details}

User profile:
- Age group: {age}
- Gender: {gender}
- Skin type: {skin_type}
- Sensitive skin: {sensitivity}
- Main concern: {concern}
- Severity: {severity}

For EACH product listed above, write:
1. "benefits": 1-2 short sentences on why this specific product suits this user, mentioning key ingredients by name.
2. "how_to_use": one short sentence on when and how to apply it (e.g. "Apply morning and night after cleansing").

Keep language simple, warm, and easy to understand. Do not recommend products outside this list.

Respond with ONLY valid JSON, no markdown formatting, no code fences, in exactly this shape:
{{"products": [{{"product_name": "...", "benefits": "...", "how_to_use": "..."}}]}}
"""


def _get_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError(
            "GEMINI_API_KEY not found. Set it in your .env file or environment variables."
        )
    return key


def _format_product_details(products: list[dict]) -> str:
    lines = []
    for i, p in enumerate(products, start=1):
        lines.append(
            f"{i}. {p['product_name']} ({p.get('product_type', 'Skin Care')})\n"
            f"   Price: {p.get('price_pkr', p.get('price'))}\n"
            f"   Ingredients: {p.get('ingredients', 'N/A')}\n"
            f"   Addresses: {p.get('primary_concern', '')}"
        )
    return "\n\n".join(lines)


def build_product_prompt(products: list[dict], user_answers: dict) -> str:
    return PRODUCT_RECOMMENDATION_PROMPT.format(
        count=len(products),
        product_details=_format_product_details(products),
        age=user_answers.get("Age", ""),
        gender=user_answers.get("Gender", ""),
        skin_type=user_answers.get("Skin_Type", ""),
        sensitivity=user_answers.get("Sensitivity", ""),
        concern=user_answers.get("Concern", ""),
        severity=user_answers.get("Severity", ""),
    )


def parse_product_recommendations(raw_text: str) -> dict:
    """
    Parse the LLM's JSON response into {product_name: {benefits, how_to_use}}.
    Returns an empty dict if parsing fails, so callers can fall back gracefully.
    """
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
        result = {}
        for item in data.get("products", []):
            name = item.get("product_name", "")
            if name:
                result[name] = {
                    "benefits": item.get("benefits", ""),
                    "how_to_use": item.get("how_to_use", ""),
                }
        return result
    except Exception:
        return {}


def get_gemini_product_recommendation(
    products: list[dict], user_answers: dict
) -> str:
    """Generate a friendly product recommendation message via Gemini."""
    api_key = _get_api_key()
    genai.configure(api_key=api_key)
    prompt = build_product_prompt(products, user_answers)
    last_error = None
    for model_name in GEMINI_MODELS:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Gemini API error: {last_error}")
