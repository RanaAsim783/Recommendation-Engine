"""
DermIQ — Analyze all products in brand_products.csv with Gemini API.

Adds suitability columns for filtering in the chatbot.
Uses 1 second delay between API calls. Groq fallback if Gemini fails.

Run: python scripts/analyze_products.py
Optional: python scripts/analyze_products.py --limit 10
"""

import argparse
import os
import sys
import time

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CSV_PATH = os.path.join(DATA_DIR, "brand_products.csv")

ANALYSIS_COLUMNS = [
    "suited_age_groups",
    "suited_genders",
    "suited_skin_types",
    "suited_sensitivity",
    "suited_climates",
    "suited_concerns",
    "suited_severity_levels",
]

sys.path.insert(0, PROJECT_ROOT)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from utils.product_analyzer import (  # noqa: E402
    ANALYSIS_DELAY_SECONDS,
    analyze_product_suitability,
    lists_to_csv_value,
)


def analyze_all(limit: int | None = None):
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"{CSV_PATH} not found. Run 'python scripts/scrape_products.py' first."
        )

    df = pd.read_csv(CSV_PATH)
    for col in ANALYSIS_COLUMNS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str).replace("nan", "")

    pending = df[
        df["suited_skin_types"].fillna("").astype(str).str.strip().isin(["", "nan"])
    ]
    if limit:
        pending = pending.head(limit)

    total = len(pending)
    print(f"Analyzing {total} products (1s delay between calls)...")

    for i, idx in enumerate(pending.index, start=1):
        row = df.loc[idx]
        name = row["product_name"]
        print(f"  [{i}/{total}] {name}")

        try:
            analysis, provider = analyze_product_suitability(
                product_name=name,
                product_type=str(row.get("product_type", "")),
                ingredients=str(row.get("ingredients", "")),
            )
            for col in ANALYSIS_COLUMNS:
                df.at[idx, col] = lists_to_csv_value(analysis[col])
            print(f"       OK via {provider}")
        except Exception as exc:
            print(f"       FAILED: {exc}", file=sys.stderr)

        df.to_csv(CSV_PATH, index=False, encoding="utf-8")
        if i < total:
            time.sleep(ANALYSIS_DELAY_SECONDS)

    print(f"\nAnalysis saved to {CSV_PATH}")


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Analyze only N products")
    args = parser.parse_args()
    try:
        analyze_all(limit=args.limit)
    except Exception as exc:
        print(f"Analysis failed: {exc}", file=sys.stderr)
        sys.exit(1)
