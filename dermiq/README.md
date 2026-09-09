# DermIQ — AI Skincare Product Recommendation Engine

DermIQ is a conversational chatbot that asks users 6 questions about their skin and recommends matching **Hemani Herbals** skincare products from a scraped and AI-analyzed catalog.

## How It Works

1. **Scrape** products from [Hemani Herbals Skin Care](https://pk.hemaniherbals.com/collections/skin-care)
2. **Analyze** each product's ingredients with Gemini 1.5 Flash (Groq fallback)
3. **Answer** — user answers 6 questions via tappable chips, or types a free-text answer for Age, Skin Type, and Sensitivity (auto-classified into the trained categories)
4. **Filter** products matching all 6 user inputs
5. **Recommend** top 3 products, each with its own AI-written benefits and how-to-use line
6. **Display** product cards in Streamlit with price, benefits, routine, and Visit Product links

## Project Structure

```
dermiq/
├── app.py                          # Streamlit chatbot + product cards (light theme, chip UI)
├── scripts/
│   ├── scrape_products.py          # Step 1: scrape → brand_products.csv
│   └── analyze_products.py         # Step 2: Gemini suitability analysis
├── data/
│   └── brand_products.csv          # Product catalog + AI analysis columns
├── utils/
│   ├── product_analyzer.py         # Gemini/Groq ingredient analysis
│   ├── product_matcher.py          # Filter products by user profile (6 fields)
│   ├── answer_classifier.py        # Free-text → category classifier (Age, Skin Type, Sensitivity)
│   ├── gemini_helper.py            # Per-product recommendation JSON (benefits + how_to_use)
│   ├── groq_helper.py              # Groq fallback
│   └── llm_router.py               # Gemini → Groq router, parses structured recommendations
├── requirements.txt
└── .env                            # API keys (not committed)
```

## Setup

```bash
cd dermiq
pip install -r requirements.txt
```

Create `.env`:
```
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
```

## Run Pipeline

```bash
# Step 1 — Scrape ~193 products from Hemani Herbals
python scripts/scrape_products.py

# Step 2 — Analyze ingredients (1s delay per product; use --limit for testing)
python scripts/analyze_products.py
# python scripts/analyze_products.py --limit 10

# Step 3 — Launch chatbot
streamlit run app.py
```

## User Profile Fields (6 questions)

The Climate question was removed from the flow. `product_matcher.py` no longer requires a climate match, so this doesn't affect result quality — it only narrows down the same 6 remaining fields.

| Field | Options | Input style |
|-------|---------|-------------|
| Age | 14-18, 19-24, 25-36, 37-45, 45+ | Free text or chips |
| Skin Type | Normal, Dry, Oily, Combination | Free text or chips |
| Sensitivity | Yes, No | Free text or chips |
| Gender | Male, Female | Chips |
| Concern | Acne, Dark Circles, Dark Spots, etc. | Chips |
| Severity | Mild, Moderate, Severe | Chips |

### How free-text answers are classified

For Age, Skin Type, and Sensitivity, the user can type a natural answer (e.g. "I'm 27", "gets shiny by midday", "sometimes it stings with new products") instead of picking an option directly.

`answer_classifier.py` handles this in two steps:
1. **Rule-based matching** — regex/keyword rules catch the common phrasings instantly, with no API call
2. **Gemini fallback** — if the rules can't confidently classify the text, it's sent to Gemini with the exact list of valid categories, and the model returns one of them (or "unclear")

If neither step produces a confident match, the UI tells the user it's not sure and asks them to pick a chip instead — it never silently guesses.

## Recommendation Format

`gemini_helper.py`'s prompt asks for a JSON response with a `benefits` and `how_to_use` line **per product**, instead of one combined paragraph for all 3 products. `llm_router.py` parses this JSON via `parse_product_recommendations()`. If parsing fails for any reason, `app.py` falls back to displaying the raw AI response as a single block, so a malformed response never breaks the page.

## License

Academic project — National University of Sciences & Technology.
