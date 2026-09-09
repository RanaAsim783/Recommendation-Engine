"""
DermIQ — Model Training Script
Loads Dataset_2.xlsx, preprocesses data, trains a Random Forest classifier,
and saves the model plus label encoders.

Run independently with: python model/train_model.py
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Paths relative to project root (dermiq/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "Dataset_2.xlsx")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
SHEET_NAME = "Clinically_Corrected_ML_Dataset"

FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "Skin_Type",
    "Sensitivity",
    "Climate",
    "Concern",
    "Severity",
]
TARGET_COLUMN = "Ingredients"
EFFECTS_COLUMN = "Effects"


def load_and_preprocess(data_path: str) -> pd.DataFrame:
    """Load the Excel dataset and apply cleaning steps."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Please place Dataset_2.xlsx in the data/ folder."
        )

    df = pd.read_excel(data_path, sheet_name=SHEET_NAME, engine="openpyxl")

    expected_columns = FEATURE_COLUMNS + [TARGET_COLUMN, EFFECTS_COLUMN]
    if list(df.columns) != expected_columns:
        df.columns = expected_columns

    df = df.drop_duplicates()
    df = df.dropna()

    for col in df.columns:
        df[col] = df[col].map(lambda x: str(x).strip().lower())

    return df


def encode_features(df: pd.DataFrame):
    """Encode categorical features and target; return encoded arrays and encoders."""
    encoders = {}
    X = np.zeros((len(df), len(FEATURE_COLUMNS)), dtype=int)

    for i, col in enumerate(FEATURE_COLUMNS):
        le = LabelEncoder()
        X[:, i] = le.fit_transform(df[col])
        encoders[col] = le

    ingredient_encoder = LabelEncoder()
    y = ingredient_encoder.fit_transform(df[TARGET_COLUMN])

    return X, y, encoders, ingredient_encoder


def build_ingredient_effects_map(df: pd.DataFrame) -> dict:
    """Map each ingredient combination to its effects explanation."""
    mapping = {}
    for _, row in df.iterrows():
        key = row[TARGET_COLUMN]
        if key not in mapping:
            mapping[key] = row[EFFECTS_COLUMN]
    return mapping


def train_and_save():
    """Main training pipeline."""
    print("Loading and preprocessing data...")
    df = load_and_preprocess(DATA_PATH)
    print(f"  Rows after cleaning: {len(df)}")

    X, y, encoders, ingredient_encoder = encode_features(df)
    ingredient_effects_map = build_ingredient_effects_map(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=15,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, "dermiq_model.pkl")
    encoders_path = os.path.join(MODEL_DIR, "dermiq_encoders.pkl")
    ingredient_encoder_path = os.path.join(MODEL_DIR, "dermiq_ingredient_encoder.pkl")
    effects_map_path = os.path.join(MODEL_DIR, "ingredient_effects_map.pkl")

    joblib.dump(model, model_path, compress=3)
    joblib.dump(encoders, encoders_path, compress=3)
    joblib.dump(ingredient_encoder, ingredient_encoder_path, compress=3)
    joblib.dump(ingredient_effects_map, effects_map_path, compress=3)

    print(f"\nModel saved to: {model_path}")
    print(f"Encoders saved to: {encoders_path}")
    print(f"Ingredient encoder saved to: {ingredient_encoder_path}")
    print(f"Effects map saved to: {effects_map_path}")
    print("\nTraining complete.")


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    try:
        train_and_save()
    except Exception as exc:
        print(f"Error during training: {exc}", file=sys.stderr)
        sys.exit(1)
