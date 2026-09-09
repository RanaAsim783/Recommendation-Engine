"""
DermIQ — Streamlit Web Application
Product-based skincare recommendation chatbot for Pakistani users.

Run with: streamlit run app.py
"""

import os
import sys

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from utils.llm_router import get_product_recommendation
from utils.product_matcher import match_products
from utils.answer_classifier import classify_free_text

st.set_page_config(
    page_title="DermIQ — Skincare Advisor",
    page_icon="favicon.png" if os.path.exists(os.path.join(PROJECT_ROOT, "favicon.png")) else "🌿",
    layout="centered",
)

NAVY = "#0A2540"
NAVY_DEEP = "#071A2E"
NAVY_SOFT = "#163656"
GOLD = "#C79A45"
GOLD_DEEP = "#9C7530"
GOLD_TINT = "#F5E9D2"
GOLD_GLOW = "#E8C77E"
EMERALD = "#2F7A5C"
ROSE = "#B85C56"
INK = "#10233A"
INK_SOFT = "#5B6B7C"
CANVAS = "#FAF6EF"
PANEL = "#F2EBDC"
LINE = "#E3D9C4"

st.markdown(
    f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
        .stApp {{ background-color: {CANVAS}; }}

        .dermiq-header {{ text-align: center; padding: 0.5rem 0 0.25rem 0; }}
        .dermiq-header h1 {{
            font-family: 'Fraunces', serif; font-weight: 600;
            color: {NAVY}; font-size: 2.1rem; margin-bottom: 0.15rem;
        }}
        .dermiq-header p {{ color: {INK_SOFT}; font-size: 0.95rem; }}

        .q-card {{
            background: #fff; border: 1.5px solid {LINE}; border-radius: 18px;
            padding: 1.6rem 1.5rem; margin: 0.9rem 0;
        }}
        .q-title {{
            font-family: 'Fraunces', serif; font-weight: 600; font-size: 1.5rem;
            color: {NAVY}; margin-bottom: 0.3rem; line-height: 1.3;
        }}
        .q-hint {{ color: {INK_SOFT}; font-size: 0.85rem; margin-bottom: 0.9rem; }}
        .or-divider {{
            text-align: center; color: {INK_SOFT}; font-size: 0.7rem; font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.08em; margin: 0.8rem 0 0.6rem;
        }}
        .match-badge-ok {{
            background: #EAF3EE; color: {EMERALD}; padding: 0.55rem 0.8rem;
            border-radius: 10px; font-size: 0.85rem; font-weight: 500; margin: 0.5rem 0;
        }}
        .match-badge-unclear {{
            background: #F7EAE8; color: {ROSE}; padding: 0.55rem 0.8rem;
            border-radius: 10px; font-size: 0.85rem; font-weight: 500; margin: 0.5rem 0;
        }}

        .bot-message {{
            background-color: {PANEL}; border-left: 4px solid {NAVY};
            padding: 0.9rem 1.1rem; border-radius: 0 12px 12px 12px;
            margin: 0.6rem 0; color: {INK}; line-height: 1.5; font-size: 0.92rem;
        }}
        .user-message {{
            background-color: #fff; border: 1.5px solid {GOLD};
            padding: 0.7rem 1.1rem; border-radius: 12px 0 12px 12px;
            margin: 0.6rem 0 0.6rem 2rem; text-align: right;
            color: {GOLD_DEEP}; font-weight: 600; font-size: 0.92rem;
        }}

        .product-card {{
            background: #fff; border: 1.5px solid {LINE}; border-radius: 16px;
            padding: 1.2rem; margin-bottom: 1rem;
        }}
        .product-price {{ color: {GOLD_DEEP}; font-size: 1.1rem; font-weight: 700; }}
        .product-benefit {{
            background: {GOLD_TINT}; border-radius: 10px; padding: 0.7rem 0.9rem;
            margin-top: 0.7rem; font-size: 0.87rem; color: {INK}; line-height: 1.5;
        }}
        .product-routine {{
            color: {NAVY_SOFT}; font-size: 0.83rem; margin-top: 0.5rem; font-weight: 500;
        }}

        /* Option chips: cream/off-white by default, shift to gold on hover, */
        /* and light up gold the instant they're pressed (the "selection" moment). */
        div[data-testid="stButton"] button {{
            border-radius: 11px !important; border: 1.5px solid {LINE} !important;
            font-weight: 500 !important; color: {NAVY} !important; background: #fff !important;
            transition: background .15s ease, border-color .15s ease, color .15s ease;
        }}
        div[data-testid="stButton"] button:hover {{
            border-color: {GOLD} !important; color: {GOLD_DEEP} !important;
            background: {GOLD_TINT} !important;
        }}
        div[data-testid="stButton"] button:active,
        div[data-testid="stButton"] button:focus:not(:focus-visible) {{
            background: {GOLD} !important; border-color: {GOLD} !important; color: {NAVY_DEEP} !important;
        }}
        div[data-testid="stButton"] button[kind="primary"] {{
            background: {GOLD} !important; border-color: {GOLD} !important; color: {NAVY_DEEP} !important;
        }}
        div[data-testid="stButton"] button[kind="primary"]:hover {{
            background: {GOLD_GLOW} !important; border-color: {GOLD_GLOW} !important;
        }}
        div[data-testid="stTextInput"] input {{
            border-radius: 11px !important; border: 1.5px solid {LINE} !important;
        }}
        div[data-testid="stTextInput"] input:focus {{
            border-color: {GOLD} !important; box-shadow: 0 0 0 1px {GOLD} !important;
        }}
        div[data-testid="stProgressBar"] > div > div {{ background-color: {NAVY}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

QUESTIONS = [
    {"key": "Age", "text": "How old are you?", "options": ["14-18", "19-24", "25-36", "37-45", "45+"],
     "hybrid": True, "placeholder": "e.g. 27", "hint": "Type your age, or just pick a range below."},
    {"key": "Gender", "text": "What is your gender?", "options": ["Male", "Female"]},
    {"key": "Skin_Type", "text": "How would you describe your skin?", "options": ["Normal", "Dry", "Oily", "Combination"],
     "hybrid": True, "placeholder": "e.g. gets shiny by midday", "hint": "Not sure of the technical term? Just describe it."},
    {"key": "Sensitivity", "text": "Does your skin get easily irritated or sensitive to products?", "options": ["Yes", "No"],
     "hybrid": True, "placeholder": "e.g. sometimes it stings with new products", "hint": "Tell us in your own words, or pick below."},
    {
        "key": "Concern",
        "text": "What is your main skin concern?",
        "options": [
            "Acne", "Dark Circles", "Dark Spots", "Dullness", "Hyperpigmentation",
            "Open Pores", "Redness", "Sun Tan", "Whiteheads / Blackheads", "Wrinkles",
        ],
    },
    {"key": "Severity", "text": "How severe is your concern?", "options": ["Mild", "Moderate", "Severe"]},
]

TOTAL_QUESTIONS = len(QUESTIONS)
GREETING = (
    "Assalam-o-Alaikum! 👋 I'm **DermIQ**, your personal skincare advisor "
    f"for Pakistan. I'll ask you {TOTAL_QUESTIONS} quick questions, then recommend the best "
    "**Hemani Herbals** products for your skin. Let's get started!"
)


def init_session_state():
    defaults = {
        "step": 0,
        "answers": {},
        "chat_history": [],
        "products": None,
        "recommendations": {},
        "recommendation_raw": None,
        "provider": None,
        "finished": False,
        "greeting_shown": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def restart():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session_state()


def add_bot_message(text: str):
    st.session_state.chat_history.append({"role": "bot", "text": text})


def add_user_message(text: str):
    st.session_state.chat_history.append({"role": "user", "text": text})


def render_chat_history():
    for msg in st.session_state.chat_history:
        css = "bot-message" if msg["role"] == "bot" else "user-message"
        st.markdown(f'<div class="{css}">{msg["text"]}</div>', unsafe_allow_html=True)


def process_answer(question_idx: int, selected: str):
    question = QUESTIONS[question_idx]
    add_user_message(selected)
    st.session_state.answers[question["key"]] = selected.lower()
    st.session_state.step = question_idx + 1

    if st.session_state.step < TOTAL_QUESTIONS:
        add_bot_message(QUESTIONS[st.session_state.step]["text"])
    else:
        add_bot_message(
            "Thank you! Searching Hemani Herbals products that match your profile... 🌿"
        )
        st.session_state.finished = True


def generate_recommendation():
    with st.spinner("Finding matching products..."):
        try:
            products = match_products(st.session_state.answers, top_n=3)
            st.session_state.products = products
        except FileNotFoundError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error(f"Product matching failed: {exc}")
            return

    if not st.session_state.products:
        st.session_state.recommendation_raw = (
            "I couldn't find products that match all your criteria exactly. "
            "Try adjusting your answers, or run `python scripts/analyze_products.py` "
            "if the product catalog hasn't been analyzed yet."
        )
        return

    with st.spinner("Generating your personalized recommendation..."):
        try:
            recommendations, provider, raw_text = get_product_recommendation(
                st.session_state.products, st.session_state.answers
            )
            st.session_state.recommendations = recommendations
            st.session_state.provider = provider
            if not recommendations:
                st.session_state.recommendation_raw = raw_text
        except RuntimeError as exc:
            st.error(str(exc))
            st.session_state.recommendation_raw = (
                "_AI explanation unavailable — product matches are shown below._"
            )


def render_product_cards(products: list[dict]):
    st.markdown("### 🛍️ Recommended Products")
    recommendations = st.session_state.recommendations or {}

    for product in products:
        col_img, col_info = st.columns([1, 2])
        with col_img:
            if product.get("image_url"):
                st.image(product["image_url"], use_container_width=True)
        with col_info:
            st.markdown(f"**{product['product_name']}**")
            st.markdown(
                f'<p class="product-price">{product.get("price_pkr", "")}</p>',
                unsafe_allow_html=True,
            )
            concern = product.get("primary_concern", st.session_state.answers.get("Concern", ""))
            st.caption(f"Addresses: **{concern.title()}**")
            st.link_button(
                "Visit Product →",
                product.get("product_url", "#"),
                use_container_width=True,
            )

        info = recommendations.get(product["product_name"])
        if info:
            benefits = info.get("benefits", "")
            how_to_use = info.get("how_to_use", "")
            st.markdown(
                f'<div class="product-benefit">✨ {benefits}'
                + (f'<div class="product-routine">🕐 {how_to_use}</div>' if how_to_use else "")
                + "</div>",
                unsafe_allow_html=True,
            )
        st.divider()

    if not recommendations and st.session_state.recommendation_raw:
        st.markdown(
            '<div class="bot-message">' + st.session_state.recommendation_raw + "</div>",
            unsafe_allow_html=True,
        )

    if st.session_state.provider == "groq":
        st.caption("ℹ️ Recommendation generated via backup AI (Groq).")


def render_chip_grid(question: dict, step: int):
    options = question["options"]
    cols_per_row = 2
    for row_start in range(0, len(options), cols_per_row):
        row_options = options[row_start:row_start + cols_per_row]
        cols = st.columns(len(row_options))
        for i, opt in enumerate(row_options):
            with cols[i]:
                if st.button(opt, key=f"chip_{step}_{row_start + i}", use_container_width=True):
                    process_answer(step, opt)
                    st.rerun()


def render_hybrid_question(question: dict, step: int):
    text_val = st.text_input(
        "", placeholder=question.get("placeholder", ""), key=f"text_{step}",
        label_visibility="collapsed",
    )
    if text_val and text_val.strip():
        matched = classify_free_text(question["key"], text_val, question["options"])
        if matched:
            st.markdown(
                f'<div class="match-badge-ok">✓ Got it — that sounds like: <b>{matched}</b></div>',
                unsafe_allow_html=True,
            )
            if st.button(f"Continue with '{matched}'", key=f"confirm_{step}", type="primary", use_container_width=True):
                process_answer(step, matched)
                st.rerun()
        else:
            st.markdown(
                '<div class="match-badge-unclear">Not quite sure — try rephrasing, or pick a chip below</div>',
                unsafe_allow_html=True,
            )
    st.markdown('<div class="or-divider">or choose one</div>', unsafe_allow_html=True)
    render_chip_grid(question, step)


init_session_state()

st.markdown(
    """
    <div class="dermiq-header">
        <h1>DermIQ</h1>
        <p>Your AI Skincare Advisor for Pakistan · Powered by Hemani Herbals</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.greeting_shown:
    add_bot_message(GREETING)
    add_bot_message(QUESTIONS[0]["text"])
    st.session_state.greeting_shown = True

progress = min(st.session_state.step / TOTAL_QUESTIONS, 1.0)
st.progress(
    progress,
    text=f"Question {min(st.session_state.step + 1, TOTAL_QUESTIONS)} of {TOTAL_QUESTIONS}",
)

render_chat_history()

if not st.session_state.finished and st.session_state.step < TOTAL_QUESTIONS:
    current = QUESTIONS[st.session_state.step]
    st.markdown('<div class="q-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="q-title">{current["text"]}</div>', unsafe_allow_html=True)
    if current.get("hint"):
        st.markdown(f'<div class="q-hint">{current["hint"]}</div>', unsafe_allow_html=True)

    if current.get("hybrid"):
        render_hybrid_question(current, st.session_state.step)
    else:
        render_chip_grid(current, st.session_state.step)
    st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.finished and st.session_state.products is None and not st.session_state.recommendations and not st.session_state.recommendation_raw:
    generate_recommendation()
    st.rerun()

if st.session_state.products:
    render_product_cards(st.session_state.products)
elif st.session_state.recommendation_raw:
    st.markdown(
        '<div class="bot-message">' + st.session_state.recommendation_raw + "</div>",
        unsafe_allow_html=True,
    )

if st.session_state.finished:
    st.divider()
    if st.button("🔄 Start Over", use_container_width=True):
        restart()
        st.rerun()
