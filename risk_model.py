from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


DATASET_PATH = Path("data/Maternal Health Risk Data Set.csv")
FEATURE_COLUMNS = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
TARGET_COLUMN = "RiskLevel"


@dataclass
class RiskPrediction:
    label: str
    probabilities: dict[str, float]
    confidence: float


@dataclass
class RiskModelBundle:
    model: RandomForestClassifier
    label_encoder: LabelEncoder
    accuracy: float
    feature_columns: list[str]
    dataset: pd.DataFrame


def normalize_risk_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset = dataset.copy()
    dataset.columns = dataset.columns.str.strip()
    dataset[TARGET_COLUMN] = dataset[TARGET_COLUMN].str.strip().str.lower()
    return dataset


def load_risk_dataset(path: str | Path | None = None) -> pd.DataFrame:
    dataset = pd.read_csv(path or DATASET_PATH)
    dataset = normalize_risk_dataset(dataset)
    return dataset


def train_risk_model_from_dataframe(dataset: pd.DataFrame) -> RiskModelBundle:
    dataset = normalize_risk_dataset(dataset)
    features = dataset[FEATURE_COLUMNS]
    targets = dataset[TARGET_COLUMN]

    label_encoder = LabelEncoder()
    encoded_targets = label_encoder.fit_transform(targets)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        encoded_targets,
        test_size=0.2,
        random_state=42,
        stratify=encoded_targets,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)

    predicted = model.predict(x_test)
    accuracy = accuracy_score(y_test, predicted)

    return RiskModelBundle(
        model=model,
        label_encoder=label_encoder,
        accuracy=accuracy,
        feature_columns=FEATURE_COLUMNS,
        dataset=dataset,
    )


@lru_cache(maxsize=1)
def train_risk_model() -> RiskModelBundle:
    return train_risk_model_from_dataframe(load_risk_dataset())


def predict_risk_with_bundle(features: dict[str, float], bundle: RiskModelBundle) -> RiskPrediction:
    row = pd.DataFrame([[features[column] for column in bundle.feature_columns]], columns=bundle.feature_columns)
    probabilities = bundle.model.predict_proba(row)[0]
    predicted_index = int(probabilities.argmax())
    predicted_label = bundle.label_encoder.inverse_transform([predicted_index])[0]

    probability_map = {
        label: float(probability)
        for label, probability in zip(bundle.label_encoder.classes_, probabilities)
    }

    return RiskPrediction(
        label=predicted_label,
        probabilities=probability_map,
        confidence=float(probabilities[predicted_index]),
    )


def predict_risk(features: dict[str, float]) -> RiskPrediction:
    return predict_risk_with_bundle(features, train_risk_model())


def get_dataset_overview_from_bundle(bundle: RiskModelBundle) -> pd.DataFrame:
    dataset = bundle.dataset.copy()
    summary = (
        dataset.groupby(TARGET_COLUMN)[FEATURE_COLUMNS]
        .mean()
        .round(2)
        .reset_index()
        .rename(columns={TARGET_COLUMN: "RiskLevel"})
    )
    return summary


def get_dataset_overview() -> pd.DataFrame:
    return get_dataset_overview_from_bundle(train_risk_model())


def build_dataset_context(question: str, bundle: RiskModelBundle | None = None) -> str:
    bundle = bundle or train_risk_model()
    dataset = bundle.dataset

    class_counts = dataset[TARGET_COLUMN].value_counts().to_dict()
    feature_ranges = dataset[FEATURE_COLUMNS].agg(["min", "max", "mean"]).round(2)
    grouped_means = get_dataset_overview_from_bundle(bundle)

    context_parts = [
        "Dataset: Maternal Health Risk Data Set.",
        f"Rows: {len(dataset)}.",
        f"Features: {', '.join(FEATURE_COLUMNS)}.",
        f"Risk level distribution: {class_counts}.",
        f"Model validation accuracy: {bundle.accuracy:.3f}.",
        f"Feature ranges and averages:\n{feature_ranges.to_string()}",
        f"Average vitals by risk level:\n{grouped_means.to_string(index=False)}",
    ]

    lowered = question.lower()
    if "sample" in lowered or "example" in lowered or "row" in lowered:
        context_parts.append(f"Sample records:\n{dataset.head(8).to_string(index=False)}")

    if "high risk" in lowered:
        high_risk = dataset[dataset[TARGET_COLUMN] == "high risk"].head(10)
        context_parts.append(f"High-risk examples:\n{high_risk.to_string(index=False)}")

    return "\n\n".join(context_parts)
