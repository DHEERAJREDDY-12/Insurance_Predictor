import argparse
import os
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None  # type: ignore

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None  # type: ignore


def infer_column_types(df: pd.DataFrame, target: str, id_cols: Optional[List[str]] = None) -> Tuple[List[str], List[str]]:
    id_cols = id_cols or []
    feature_cols = [c for c in df.columns if c not in id_cols + [target]]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])
    return preprocessor


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def get_models(random_state: int = 42):
    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=400, max_depth=None, n_jobs=-1, random_state=random_state),
    }
    if XGBRegressor is not None:
        models["xgboost"] = XGBRegressor(
            n_estimators=800,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
        )
    if LGBMRegressor is not None:
        models["lightgbm"] = LGBMRegressor(
            n_estimators=1200,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
        )
    return models


def main():
    parser = argparse.ArgumentParser(description="Train models for title insurance premium prediction")
    parser.add_argument("--data-path", default="insurance.csv", help="Path to CSV dataset")
    parser.add_argument("--target", default="charges", help="Target column name (e.g., charges)")
    parser.add_argument("--id-cols", nargs="*", default=None, help="Optional ID columns to drop")
    parser.add_argument("--valid-size", type=float, default=0.2, help="Validation split size")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-out", default="artifacts/best_model.joblib")
    parser.add_argument("--metrics-out", default="artifacts/metrics.json")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    df = pd.read_csv(args.data_path)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in data. Columns: {list(df.columns)}")

    numeric_cols, categorical_cols = infer_column_types(df, args.target, args.id_cols)

    X = df[numeric_cols + categorical_cols]
    y = df[args.target].values

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=args.valid_size, random_state=args.random_state
    )

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    models = get_models(args.random_state)

    best_name = None
    best_model = None
    best_metrics = None

    for name, estimator in models.items():
        model = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("estimator", estimator),
        ])
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        metrics = evaluate(y_valid, preds)
        print(f"Model: {name} => RMSE={metrics['rmse']:.4f} MAE={metrics['mae']:.4f} R2={metrics['r2']:.4f}")
        if best_metrics is None or metrics["rmse"] < best_metrics["rmse"]:
            best_name = name
            best_model = model
            best_metrics = metrics

    assert best_model is not None and best_metrics is not None and best_name is not None

    joblib.dump(best_model, args.model_out)

    try:
        import json
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump({"best_model": best_name, **best_metrics}, f, indent=2)
    except Exception:
        pass

    print(f"Saved best model '{best_name}' to {args.model_out}")


if __name__ == "__main__":
    main()
