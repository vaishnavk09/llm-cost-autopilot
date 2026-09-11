from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from app.features import FEATURE_NAMES, feature_vector
from app.settings import abs_path, get_settings

LABELS = {1: "simple", 2: "moderate", 3: "complex"}


class ComplexityClassifier:
    def __init__(self, model: RandomForestClassifier | None = None) -> None:
        self.model = model or RandomForestClassifier(
            n_estimators=180,
            max_depth=8,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced",
        )

    def predict(self, prompt: str) -> int:
        vec = [feature_vector(prompt)]
        return int(self.model.predict(vec)[0])

    def predict_proba(self, prompt: str) -> dict[int, float]:
        proba = self.model.predict_proba([feature_vector(prompt)])[0]
        return {int(cls): float(p) for cls, p in zip(self.model.classes_, proba)}

    def save(self, path: str | Path | None = None) -> Path:
        dest = Path(path) if path else abs_path(get_settings().classifier_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "features": FEATURE_NAMES}, dest)
        return dest

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ComplexityClassifier":
        dest = Path(path) if path else abs_path(get_settings().classifier_path)
        payload = joblib.load(dest)
        return cls(model=payload["model"])


def train_from_rows(
    prompts: list[str],
    labels: list[int],
    test_size: float = 0.2,
) -> tuple[ComplexityClassifier, dict]:
    X = [feature_vector(p) for p in prompts]
    y = labels
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    clf = ComplexityClassifier()
    clf.model.fit(X_train, y_train)
    preds = clf.model.predict(X_test)
    acc = float(accuracy_score(y_test, preds))
    cm = confusion_matrix(y_test, preds, labels=[1, 2, 3]).tolist()
    metrics = {
        "accuracy": acc,
        "confusion_matrix": cm,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_names": FEATURE_NAMES,
    }
    return clf, metrics
