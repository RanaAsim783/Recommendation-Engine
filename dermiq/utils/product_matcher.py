"""
DermIQ — Product matching from brand_products.csv
Filters products that match all 6 user profile inputs.
"""

import os

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "brand_products.csv")

USER_TO_COLUMN = {
    "Age": "suited_age_groups",
    "Gender": "suited_genders",
    "Skin_Type": "suited_skin_types",
    "Sensitivity": "suited_sensitivity",
    "Concern": "suited_concerns",
    "Severity": "suited_severity_levels",
}


def _normalize(value: str) -> str:
    return str(value).strip().lower()


def _parse_pipe_list(value) -> set[str]:
    if pd.isna(value) or not str(value).strip():
        return set()
    return {_normalize(v) for v in str(value).split("|") if v.strip()}


def _row_matches(row: pd.Series, user_answers: dict) -> bool:
    for user_key, col in USER_TO_COLUMN.items():
        user_val = _normalize(user_answers[user_key])
        suited = _parse_pipe_list(row.get(col, ""))
        if not suited:
            return False
        if user_val not in suited:
            return False
    return True


def load_products() -> pd.DataFrame:
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Product catalog not found at {CSV_PATH}. "
            "Run scrape_products.py and analyze_products.py first."
        )
    return pd.read_csv(CSV_PATH)


def match_products(user_answers: dict, top_n: int = 3) -> list[dict]:
    """
    Return up to top_n products matching all 6 user inputs.
    """
    df = load_products()
    analyzed = df[df["suited_skin_types"].fillna("").astype(str).str.strip() != ""]
    matches = analyzed[analyzed.apply(lambda row: _row_matches(row, user_answers), axis=1)]

    results = []
    for _, row in matches.head(top_n).iterrows():
        concerns = _parse_pipe_list(row.get("suited_concerns", ""))
        primary_concern = user_answers.get("Concern", "")
        if _normalize(primary_concern) not in concerns and concerns:
            display_concern = next(iter(concerns)).title()
        else:
            display_concern = primary_concern

        price = row.get("price_pkr") or f"Rs.{float(row.get('price', 0)):,.2f}"
        results.append(
            {
                "product_name": row["product_name"],
                "ingredients": row.get("ingredients", ""),
                "product_type": row.get("product_type", ""),
                "price": row.get("price", 0),
                "price_pkr": price,
                "image_url": row.get("image_url", ""),
                "product_url": row.get("product_url", ""),
                "primary_concern": display_concern,
            }
        )
    return results
