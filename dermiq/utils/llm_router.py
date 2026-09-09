"""
DermIQ — LLM Recommendation Router
Tries Gemini first, falls back to Groq on failure.
"""

from utils.gemini_helper import (
    get_gemini_product_recommendation,
    parse_product_recommendations,
)
from utils.groq_helper import get_groq_product_recommendation


def get_product_recommendation(
    products: list[dict], user_answers: dict
) -> tuple[dict, str, str]:
    """
    Format a product recommendation using Gemini, with Groq as fallback.

    Returns:
        (recommendations, provider, raw_text) where:
        - recommendations is {product_name: {benefits, how_to_use}}, empty if parsing failed
        - provider is 'gemini' or 'groq'
        - raw_text is the unparsed response, used as a fallback display if parsing failed
    """
    try:
        raw_text = get_gemini_product_recommendation(products, user_answers)
        provider = "gemini"
    except Exception:
        try:
            raw_text = get_groq_product_recommendation(products, user_answers)
            provider = "groq"
        except Exception as groq_exc:
            raise RuntimeError(
                "Both Gemini and Groq APIs failed. "
                "Please check your API keys and try again."
            ) from groq_exc

    recommendations = parse_product_recommendations(raw_text)
    return recommendations, provider, raw_text
