"""
DermIQ — Groq API Fallback Integration
Used when Gemini API fails or rate limits are exceeded.
"""

import os

from dotenv import load_dotenv
from groq import Groq

from utils.gemini_helper import build_product_prompt

load_dotenv()

GROQ_MODEL = "llama-3.1-8b-instant"


def _get_api_key() -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError(
            "GROQ_API_KEY not found. Set it in your .env file or environment variables."
        )
    return key


def get_groq_product_recommendation(
    products: list[dict], user_answers: dict
) -> str:
    """Generate a friendly product recommendation message via Groq."""
    try:
        api_key = _get_api_key()
        client = Groq(api_key=api_key)
        prompt = build_product_prompt(products, user_answers)

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        "Please write my personalized product recommendation "
                        "based on the matched products above."
                    ),
                },
            ],
            temperature=0.7,
            max_tokens=1500,
        )

        if not response.choices or not response.choices[0].message.content:
            raise RuntimeError("Groq returned an empty response.")

        return response.choices[0].message.content.strip()
    except Exception as exc:
        raise RuntimeError(f"Groq API error: {exc}") from exc
