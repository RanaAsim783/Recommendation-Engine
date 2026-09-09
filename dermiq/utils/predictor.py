"""
DermIQ — Prediction Module
Loads the trained model and encoders, encodes user inputs, and returns
predicted ingredients with their effects.
"""

import os

import joblib
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")

FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "Skin_Type",
    "Sensitivity",
    "Climate",
    "Concern",
    "Severity",
]

_model = None
_encoders = None
_ingredient_encoder = None
_effects_map = None


def _load_artifacts():
    """Lazy-load model artifacts on first use."""
    global _model, _encoders, _ingredient_encoder, _effects_map

    if _model is not None:
        return

    model_path = os.path.join(MODEL_DIR, "dermiq_model.pkl")
    encoders_path = os.path.join(MODEL_DIR, "dermiq_encoders.pkl")
    ingredient_encoder_path = os.path.join(MODEL_DIR, "dermiq_ingredient_encoder.pkl")
    effects_map_path = os.path.join(MODEL_DIR, "ingredient_effects_map.pkl")

    for path in (model_path, encoders_path, ingredient_encoder_path, effects_map_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model artifact not found: {path}. "
                "Run 'python model/train_model.py' first."
            )

    _model = joblib.load(model_path)
    _encoders = joblib.load(encoders_path)
    _ingredient_encoder = joblib.load(ingredient_encoder_path)
    _effects_map = joblib.load(effects_map_path)


def normalize_input(value: str) -> str:
    """Standardize user input to lowercase stripped string."""
    return str(value).strip().lower()


def encode_user_inputs(user_answers: dict) -> np.ndarray:
    """
    Encode the 7 user answers into a feature vector for the model.

    user_answers keys must match FEATURE_COLUMNS.
    Values are normalized to lowercase before encoding.
    """
    _load_artifacts()

    encoded = []
    for col in FEATURE_COLUMNS:
        raw = normalize_input(user_answers[col])
        le = _encoders[col]
        if raw not in le.classes_:
            raise ValueError(
                f"Invalid value '{user_answers[col]}' for {col}. "
                f"Expected one of: {list(le.classes_)}"
            )
        encoded.append(le.transform([raw])[0])

    return np.array([encoded])


def predict_ingredients(user_answers: dict) -> tuple[str, str]:
    """
    Predict recommended ingredients and look up their effects.

    Returns:
        (ingredients, effects) as strings ready for the LLM prompt.
    """
    _load_artifacts()

    features = encode_user_inputs(user_answers)
    prediction_idx = _model.predict(features)[0]
    ingredients = _ingredient_encoder.inverse_transform([prediction_idx])[0]

    effects = _effects_map.get(ingredients, "Effects information not available.")
    return ingredients, effects
