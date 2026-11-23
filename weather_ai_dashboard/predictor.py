import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd


class WeatherPredictor:
    def __init__(self, artifacts_dir: str | Path = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        model_path = self.artifacts_dir / "model.joblib"
        bounds_path = self.artifacts_dir / "feature_bounds.json"
        metrics_path = self.artifacts_dir / "metrics.json"

        if not model_path.exists() or not bounds_path.exists():
            raise FileNotFoundError("Артефакты модели не найдены. Сначала запустите train.py")

        self.model = joblib.load(model_path)
        with open(bounds_path, "r", encoding="utf-8") as f:
            self.bounds = json.load(f)

        self.feature_importances = {}
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                self.feature_importances = json.load(f).get("feature_importances", {})

    def validate_and_predict(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        warnings: List[str] = []
        cleaned_row: Dict[str, Any] = {}

        for col, stats in self.bounds["numeric"].items():
            if col not in user_input or user_input[col] is None or str(user_input[col]).strip() == "":
                raise ValueError(f"Поле '{col}' не может быть пустым.")
            try:
                val = float(user_input[col])
            except (TypeError, ValueError):
                raise ValueError(f"Поле '{col}' должно быть числом.")

            if val < stats["min"] or val > stats["max"]:
                warnings.append(
                    f"Значение '{col}' ({val}) выходит за диапазон обучающих данных "
                    f"[{stats['min']:.1f}; {stats['max']:.1f}]. Прогноз может быть менее точным."
                )
            cleaned_row[col] = val

        for col, allowed in self.bounds["categorical"].items():
            if col not in user_input or not user_input[col]:
                raise ValueError(f"Не выбрано значение для '{col}'.")
            val = str(user_input[col])
            if val not in allowed:
                warnings.append(f"Неизвестная категория '{val}' для признака '{col}'.")
            cleaned_row[col] = val

        df_input = pd.DataFrame([cleaned_row])
        pred_class = str(self.model.predict(df_input)[0])
        proba_values = self.model.predict_proba(df_input)[0]

        probabilities = {
            str(cls): round(float(prob), 4)
            for cls, prob in sorted(zip(self.model.classes_, proba_values), key=lambda x: x[1], reverse=True)
        }

        return {
            "prediction": pred_class,
            "confidence": probabilities[pred_class],
            "probabilities": probabilities,
            "warnings": warnings,
            "top_features": list(self.feature_importances.items())[:5],
        }
