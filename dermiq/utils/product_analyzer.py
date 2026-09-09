"""
DermIQ — Product suitability analysis via Gemini (Groq fallback).
"""

import json
import os
import re
import time

import google.generativeai as genai
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GEMINI_MODELS = ["gemini-1.5-flash", "gemini-2.5-flash", "gemini-2.0-flash"]
GROQ_MODEL = "llama-3.1-8b-instant"
ANALYSIS_DELAY_SECONDS = 1.0

ANALYSIS_PROMPT = """You are a dermatology-aware skincare analyst for Pakistani users.

Analyze this product based on its ingredient list and product type.

Product name: {product_name}
Product type: {product_type}
Ingredients: {ingredients}

Return ONLY valid JSON (no markdown) with these exact keys. Use lowercase values.
Each list must contain ONLY values from the allowed options below.

{{
  "suited_age_groups": [],
  "suited_genders": [],
  "suited_skin_types": [],
  "suited_sensitivity": [],
  "suited_climates": [],
  "suited_concerns": [],
  "suited_severity_levels": []
}}

Allowed options:
- suited_age_groups: 14-18, 19-24, 25-36, 37-45, 45+
- suited_genders: male, female
- suited_skin_types: normal, dry, oily, combination
- suited_sensitivity: yes, no
- suited_climates: hot and humid (e.g., karachi), dry and hot, cold and dry
- suited_concerns: acne, dark circles, dark spots, dullness, hyperpigmentation, open pores, redness, sun tan, whiteheads / blackheads, wrinkles
- suited_severity_levels: mild, moderate, severe

Include all options that reasonably apply. Be inclusive but accurate.
"""


def _parse_json_response(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def _normalize_list(values) -> list[str]:
    if not values:
        return []
    if isinstance(values, str):
        values = [v.strip() for v in re.split(r"[,;|]", values) if v.strip()]
    return [str(v).strip().lower() for v in values]


def _call_gemini(prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set.")
    genai.configure(api_key=api_key)
    last_error = None
    for model_name in GEMINI_MODELS:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"All Gemini models failed: {last_error}")


def _call_groq(prompt: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set.")
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Groq returned empty response.")
    return content.strip()


def _heuristic_analysis(product_name: str, product_type: str, ingredients: str) -> dict:
    """Keyword-based fallback when API calls fail."""
    text = f"{product_name} {product_type} {ingredients}".lower()

    skin_types = {"normal"}
    concerns = set()
    severity = {"mild", "moderate"}
    climates = {"hot and humid (e.g., karachi)", "dry and hot", "cold and dry"}
    sensitivity = {"yes", "no"}

    if any(k in text for k in ("neem", "salicylic", "bha", "acne", "tea tree")):
        concerns.add("acne")
        concerns.add("whiteheads / blackheads")
        skin_types.update({"oily", "combination"})
    if any(k in text for k in ("bright", "saffron", "vitamin c", "ubtan", "glow", "licorice")):
        concerns.update({"dullness", "hyperpigmentation", "dark spots"})
    if any(k in text for k in ("rose", "aloe", "hyaluronic", "hydra", "moistur")):
        concerns.add("dullness")
        skin_types.update({"dry", "normal", "combination"})
    if any(k in text for k in ("sun", "spf", "sunscreen")):
        concerns.add("sun tan")
    if any(k in text for k in ("oil control", "lemon", "charcoal")):
        skin_types.update({"oily", "combination"})
        concerns.add("open pores")
    if any(k in text for k in ("retinol", "collagen", "anti aging", "wrinkle")):
        concerns.add("wrinkles")
        severity.add("severe")
    if any(k in text for k in ("centella", "chamomile", "sooth", "redness")):
        concerns.add("redness")
        sensitivity.add("yes")
    if not concerns:
        concerns.update({"dullness", "acne"})

    return {
        "suited_age_groups": ["14-18", "19-24", "25-36", "37-45", "45+"],
        "suited_genders": ["male", "female"],
        "suited_skin_types": sorted(skin_types),
        "suited_sensitivity": sorted(sensitivity),
        "suited_climates": sorted(climates),
        "suited_concerns": sorted(concerns),
        "suited_severity_levels": sorted(severity),
    }


def analyze_product_suitability(
    product_name: str,
    product_type: str,
    ingredients: str,
) -> tuple[dict, str]:
    """
    Analyze product suitability. Returns (analysis_dict, provider).
    provider is 'gemini' or 'groq'.
    """
    prompt = ANALYSIS_PROMPT.format(
        product_name=product_name,
        product_type=product_type,
        ingredients=ingredients,
    )

    provider = "gemini"
    try:
        text = _call_gemini(prompt)
        parsed = _parse_json_response(text)
        result = {
            "suited_age_groups": _normalize_list(parsed.get("suited_age_groups")),
            "suited_genders": _normalize_list(parsed.get("suited_genders")),
            "suited_skin_types": _normalize_list(parsed.get("suited_skin_types")),
            "suited_sensitivity": _normalize_list(parsed.get("suited_sensitivity")),
            "suited_climates": _normalize_list(parsed.get("suited_climates")),
            "suited_concerns": _normalize_list(parsed.get("suited_concerns")),
            "suited_severity_levels": _normalize_list(parsed.get("suited_severity_levels")),
        }
        return result, provider
    except Exception:
        try:
            text = _call_groq(prompt)
            provider = "groq"
            parsed = _parse_json_response(text)
            result = {
                "suited_age_groups": _normalize_list(parsed.get("suited_age_groups")),
                "suited_genders": _normalize_list(parsed.get("suited_genders")),
                "suited_skin_types": _normalize_list(parsed.get("suited_skin_types")),
                "suited_sensitivity": _normalize_list(parsed.get("suited_sensitivity")),
                "suited_climates": _normalize_list(parsed.get("suited_climates")),
                "suited_concerns": _normalize_list(parsed.get("suited_concerns")),
                "suited_severity_levels": _normalize_list(parsed.get("suited_severity_levels")),
            }
            return result, provider
        except Exception:
            return _heuristic_analysis(product_name, product_type, ingredients), "heuristic"


def lists_to_csv_value(values: list[str]) -> str:
    return "|".join(values)
