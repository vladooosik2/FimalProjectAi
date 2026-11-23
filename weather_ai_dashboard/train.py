import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = Path("data/weather_classification_data.csv")
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

NUM_FEATURES = [
    "Temperature",
    "Humidity",
    "Wind Speed",
    "Precipitation (%)",
    "Atmospheric Pressure",
    "UV Index",
    "Visibility (km)",
]
CAT_FEATURES = ["Cloud Cover", "Season", "Location"]
TARGET_COL = "Weather Type"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Файл {DATA_PATH} не найден. Скачайте 'Weather Type Classification' с Kaggle."
        )

    df = pd.read_csv(DATA_PATH)
    required = NUM_FEATURES + CAT_FEATURES + [TARGET_COL]
    missing_columns = [col for col in required if col not in df.columns]
    if missing_columns:
        raise ValueError(f"В датасете отсутствуют столбцы: {missing_columns}")

    print(f"Загружено записей: {len(df)}, признаков: {df.shape[1] - 1}")

    bounds = {"numeric": {}, "categorical": {}, "classes": sorted(df[TARGET_COL].dropna().unique().tolist())}
    for col in NUM_FEATURES:
        bounds["numeric"][col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "median": float(df[col].median()),
            "q05": float(df[col].quantile(0.05)),
            "q95": float(df[col].quantile(0.95)),
        }
    for col in CAT_FEATURES:
        bounds["categorical"][col] = sorted(df[col].dropna().unique().tolist())

    with open(ARTIFACTS_DIR / "feature_bounds.json", "w", encoding="utf-8") as f:
        json.dump(bounds, f, ensure_ascii=False, indent=2)

    df.sample(min(2000, len(df)), random_state=42).to_csv(ARTIFACTS_DIR / "eda_data.csv", index=False)

    X = df[NUM_FEATURES + CAT_FEATURES]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUM_FEATURES),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), CAT_FEATURES),
        ]
    )

    models = {
        "Baseline (Most Frequent)": DummyClassifier(strategy="most_frequent"),
        "Perceptron": Perceptron(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
    }

    comparison_results = []
    tree_importances = {}

    for name, clf in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", clf)])
        cv_acc = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy").mean()
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        comparison_results.append({
            "Модель": name,
            "CV Accuracy": round(float(cv_acc), 4),
            "Test Accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "Test F1-macro": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        })

        if name == "Decision Tree":
            feature_names = pipe.named_steps["prep"].get_feature_names_out()
            importances = pipe.named_steps["model"].feature_importances_
            agg = {col: 0.0 for col in NUM_FEATURES + CAT_FEATURES}
            for fname, imp in zip(feature_names, importances):
                clean_name = fname.split("__", 1)[1]
                for orig_col in agg:
                    if clean_name == orig_col or clean_name.startswith(orig_col + "_"):
                        agg[orig_col] += float(imp)
            tree_importances = dict(sorted(agg.items(), key=lambda x: x[1], reverse=True))

    mlp_pipe = Pipeline([
        ("prep", preprocessor),
        ("model", MLPClassifier(max_iter=300, early_stopping=True, random_state=42)),
    ])

    param_grid = {
        "model__hidden_layer_sizes": [(32,), (64, 32)],
        "model__activation": ["relu", "tanh"],
        "model__alpha": [0.0001, 0.01],
    }

    grid = GridSearchCV(mlp_pipe, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_mlp = grid.best_estimator_
    y_pred_mlp = best_mlp.predict(X_test)

    comparison_results.append({
        "Модель": f"MLPClassifier (best: {grid.best_params_['model__hidden_layer_sizes']}, {grid.best_params_['model__activation']})",
        "CV Accuracy": round(float(grid.best_score_), 4),
        "Test Accuracy": round(float(accuracy_score(y_test, y_pred_mlp)), 4),
        "Test F1-macro": round(float(f1_score(y_test, y_pred_mlp, average="macro")), 4),
    })

    labels = bounds["classes"]
    cm = confusion_matrix(y_test, y_pred_mlp, labels=labels)
    error_pairs = []
    for i, true_cls in enumerate(labels):
        for j, pred_cls in enumerate(labels):
            if i != j and cm[i, j] > 0:
                error_pairs.append({"Истинный класс": true_cls, "Предсказано": pred_cls, "Количество ошибок": int(cm[i, j])})
    error_pairs.sort(key=lambda x: x["Количество ошибок"], reverse=True)

    learning_curve_data = []
    for frac in [0.1, 0.25, 0.5, 0.75, 1.0]:
        n_samples = int(len(X_train) * frac)
        sub_pipe = Pipeline([
            ("prep", preprocessor),
            ("model", MLPClassifier(
                hidden_layer_sizes=grid.best_params_["model__hidden_layer_sizes"],
                activation=grid.best_params_["model__activation"],
                alpha=grid.best_params_["model__alpha"],
                max_iter=300,
                early_stopping=True,
                random_state=42,
            )),
        ])
        sub_pipe.fit(X_train.iloc[:n_samples], y_train.iloc[:n_samples])
        acc = accuracy_score(y_test, sub_pipe.predict(X_test))
        learning_curve_data.append({"Записей в обучении": n_samples, "Test Accuracy": round(float(acc), 4)})

    joblib.dump(best_mlp, ARTIFACTS_DIR / "model.joblib")

    metrics_payload = {
        "dataset_info": {
            "total_rows": int(len(df)),
            "missing_values": int(df.isna().sum().sum()),
            "class_distribution": df[TARGET_COL].value_counts().to_dict(),
        },
        "comparison": comparison_results,
        "best_params": grid.best_params_,
        "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
        "top_errors": error_pairs[:5],
        "feature_importances": tree_importances,
        "learning_curve": learning_curve_data,
    }

    with open(ARTIFACTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    print("Обучение завершено. Лучшие параметры:", grid.best_params_)
    print(pd.DataFrame(comparison_results).to_string(index=False))


if __name__ == "__main__":
    main()
