"""
DermIQ — Free-text answer classifier
Maps a user's own words (e.g. "I'm 27" or "gets shiny by midday") to one of
the fixed categories the trained model expects. Tries fast keyword rules
first; falls back to Gemini for anything ambiguous. Returns None if neither
approach can confidently determine an answer, so the UI can ask the person
to just pick a chip instead.
"""

import os
import re

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


def _classify_age(text: str):
    match = re.search(r"\d{1,2}", text)
    if match:
        age = int(match.group())
        if age <= 18:
            return "14-18"
        if age <= 24:
            return "19-24"
        if age <= 36:
            return "25-36"
        if age <= 45:
            return "37-45"
        return "45+"
    t = text.lower()
    if "teen" in t:
        return "14-18"
    if "young adult" in t or "early twenties" in t or "early 20" in t:
        return "19-24"
    if "middle" in t:
        return "37-45"
    if "senior" in t or "older" in t:
        return "45+"
    return None


def _classify_skin_type(text: str):
    t = text.lower()
    if re.search(r"oily|greasy|shine|shiny", t):
        return "Oily"
    if re.search(r"\bdry\b|flak|tight", t):
        return "Dry"
    if re.search(r"combo|combination|t-zone|tzone|both", t):
        return "Combination"
    if re.search(r"normal|balanced|fine\b", t):
        return "Normal"
    return None


def _classify_sensitivity(text: str):
    t = text.lower()
    if re.search(r"\byes\b|sometimes|often|react|sting|irritat|allerg|breakout|burn", t):
        return "Yes"
    if re.search(r"\bno\b|never|doesn'?t|dont|rarely", t):
        return "No"
    return None


_RULE_CLASSIFIERS = {
    "Age": _classify_age,
    "Skin_Type": _classify_skin_type,
    "Sensitivity": _classify_sensitivity,
}


def _classify_with_ai(field: str, text: str, options: list[str]):
    """Ask Gemini to map ambiguous free text onto one of the fixed options."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            f"A skincare quiz user was asked about their {field.replace('_', ' ').lower()}. "
            f'They answered in their own words: "{text}". '
            f"Classify this into EXACTLY one of these options: {options}. "
            "Reply with only the exact option text and nothing else. "
            "If it truly cannot be determined, reply with UNCLEAR."
        )
        response = model.generate_content(prompt)
        result = (response.text or "").strip()
        if result in options:
            return result
        return None
    except Exception:
        return None


def classify_free_text(field: str, text: str, options: list[str]):
    """
    Try to map a free-text answer to one of `options`.
    Returns the matched option, or None if no confident match was found.
    """
    if not text or not text.strip():
        return None

    rule_fn = _RULE_CLASSIFIERS.get(field)
    if rule_fn:
        result = rule_fn(text)
        if result:
            return result

    return _classify_with_ai(field, text, options)
