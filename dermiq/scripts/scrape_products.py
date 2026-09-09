"""
DermIQ — Scrape Hemani Herbals skin-care products into brand_products.csv.

Run: python scripts/scrape_products.py
"""

import os
import re
import sys
import time
from html import unescape
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_CSV = os.path.join(DATA_DIR, "brand_products.csv")

BASE_URL = "https://pk.hemaniherbals.com"
COLLECTION_URL = f"{BASE_URL}/collections/skin-care"
PRODUCTS_JSON = f"{COLLECTION_URL}/products.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


def extract_ingredients(body_html: str, tags: list[str], title: str) -> str:
    """Pull ingredient list from product HTML description."""
    if not body_html:
        return ", ".join(tags) if tags else title

    soup = BeautifulSoup(body_html, "html.parser")
    ingredients = []

    for heading in soup.find_all(["h2", "h3", "h4", "strong", "b"]):
        text = heading.get_text(strip=True).lower()
        if "ingredient" in text:
            sibling = heading.find_next(["ul", "ol", "p"])
            if sibling and sibling.name in ("ul", "ol"):
                for li in sibling.find_all("li"):
                    item = li.get_text(" ", strip=True)
                    if item:
                        ingredients.append(item)
            elif sibling and sibling.name == "p":
                item = sibling.get_text(" ", strip=True)
                if item:
                    ingredients.append(item)

    if not ingredients:
        plain = unescape(re.sub(r"<[^>]+>", " ", body_html))
        match = re.search(
            r"key ingredients?\s*:?\s*(.+?)(?:how to use|why you|our standards|$)",
            plain,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            block = match.group(1).strip()
            parts = re.split(r"[,;\n•\-–]+", block)
            ingredients = [p.strip() for p in parts if len(p.strip()) > 2]

    if not ingredients and tags:
        ingredients = list(tags)

    return "; ".join(dict.fromkeys(ingredients))


def infer_product_type(title: str, product_type: str, tags: list[str]) -> str:
    """Derive a readable product category from title, tags, and Shopify type."""
    combined = f"{title} {' '.join(tags)} {product_type}".lower()
    type_map = [
        ("face wash", "Face Wash"),
        ("foaming face wash", "Foaming Face Wash"),
        ("cleansing milk", "Cleansing Milk"),
        ("sunscreen", "Sunscreen"),
        ("sun block", "Sunscreen"),
        ("serum", "Serum"),
        ("cream", "Cream"),
        ("lotion", "Lotion"),
        ("gel", "Gel"),
        ("mist", "Face Mist"),
        ("toner", "Toner"),
        ("rose water", "Toner"),
        ("scrub", "Scrub"),
        ("mask", "Mask"),
        ("soap", "Soap"),
        ("lip balm", "Lip Care"),
        ("chapstick", "Lip Care"),
    ]
    for keyword, label in type_map:
        if keyword in combined:
            return label
    return product_type or "Skin Care"


def fetch_all_products() -> list[dict]:
    """Paginate Shopify collection JSON API."""
    products = []
    page = 1

    while True:
        resp = requests.get(
            PRODUCTS_JSON,
            params={"limit": 250, "page": page},
            headers=HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        batch = resp.json().get("products", [])
        if not batch:
            break
        products.extend(batch)
        print(f"  Fetched page {page}: {len(batch)} products (total {len(products)})")
        if len(batch) < 250:
            break
        page += 1
        time.sleep(0.3)

    return products


def parse_product(raw: dict) -> dict:
    """Convert Shopify product JSON to CSV row."""
    title = raw.get("title", "").strip()
    handle = raw.get("handle", "")
    tags = raw.get("tags") or []
    body_html = raw.get("body_html") or ""

    variants = raw.get("variants") or [{}]
    price = variants[0].get("price", "0")

    images = raw.get("images") or []
    image_url = images[0].get("src", "") if images else ""

    return {
        "product_name": title,
        "ingredients": extract_ingredients(body_html, tags, title),
        "product_type": infer_product_type(title, raw.get("product_type", ""), tags),
        "price": float(price),
        "price_pkr": f"Rs.{float(price):,.2f}",
        "image_url": image_url,
        "product_url": urljoin(BASE_URL, f"/products/{handle}"),
    }


def scrape_and_save():
    print(f"Scraping {COLLECTION_URL} ...")
    raw_products = fetch_all_products()
    if not raw_products:
        raise RuntimeError("No products found. Check the collection URL.")

    rows = [parse_product(p) for p in raw_products]
    df = pd.DataFrame(rows)

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\nSaved {len(df)} products to {OUTPUT_CSV}")


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    try:
        scrape_and_save()
    except Exception as exc:
        print(f"Scrape failed: {exc}", file=sys.stderr)
        sys.exit(1)
