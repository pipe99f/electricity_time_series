"""Lasso para el pico mensual de consumo (Países Bajos).

Modelo de regresión lineal con regularización L1 evaluado con validación
walk-forward sobre 2024, siguiendo la misma metodología que ARIMA, VAR y XGBoost.

Referencia: Suriya & Agusthiyar (2026), ZANCO J. Pure Appl. Sci. 38(2).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import MinMaxScaler

from common import (
    chdir_root, load_gold, split_by_year,
    compute_metrics, save_metrics, plot_forecast,
    ASSETS, TEST_YEAR,
)

COLS = ["peak_load", "avg_temperature", "humidity", "urban_population"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ingeniería de características alineada al artículo."""
    df = df.copy()

    # Variables cíclicas
    df["mes_seno"] = np.sin(2 * np.pi * df.index.month / 12)
    df["mes_coseno"] = np.cos(2 * np.pi * df.index.month / 12)

    # Lags del peak_load
    df["lag_1"] = df["peak_load"].shift(1)
    df["lag_3"] = df["peak_load"].shift(3)
    df["lag_6"] = df["peak_load"].shift(6)

    # Media móvil
    df["rolling_3_mean"] = df["peak_load"].rolling(window=3).mean()

    # Interacción climática
    df["temp_humedad"] = df["avg_temperature"] * df["humidity"]

    return df.dropna()


def run_lasso() -> dict:
    """Pipeline completo de Lasso con walk-forward."""

    # 1) Carga y feature engineering
    df = load_gold(cols=COLS).astype(float)
    df_feat = build_features(df)
    print(f"[1] Dataset con features: {len(df_feat)} obs, {df_feat.shape[1]} columnas")

    # 2) Definir features
    feature_cols = [c for c in df_feat.columns if c != "peak_load"]

    # 3) Split temporal
    train_df, test_df = split_by_year(df_feat)
    print(f"[2] Split: train={len(train_df)} obs, test={len(test_df)} obs")

    # 4) Walk-forward h=1
    print(f"[3] Walk-forward h=1 sobre {TEST_YEAR} ({len(test_df)} meses)...")
    history_df = train_df.copy()
    preds = []

    for i in range(len(test_df)):
        # Normalizar solo con datos de entrenamiento del paso actual
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(history_df[feature_cols])
        y_train = history_df["peak_load"].values

        X_test_row = scaler.transform(test_df[feature_cols].iloc[[i]])

        # Entrenar Lasso con CV para selección automática de alpha
        model = LassoCV(
            alphas=np.logspace(-4, 2, 50),
            cv=min(5, len(history_df) - 1),
            max_iter=10000,
            random_state=42,
        )
        model.fit(X_train, y_train)
        pred = float(model.predict(X_test_row)[0])
        preds.append(pred)

        # Expandir historial
        history_df = pd.concat([history_df, test_df.iloc[[i]]])

    preds_series = pd.Series(preds, index=test_df.index, name="pred")

    # 5) Métricas
    m = compute_metrics(test_df["peak_load"], preds_series)
    print(f"[4] MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  MAPE={m['MAPE']:.2f}%")
    save_metrics("lasso", m)

    # 6) Gráfico
    plot_forecast(
        df_feat["peak_load"], preds_series, test_df.index, None, None,
        f"Lasso - Pronóstico del pico mensual (Países Bajos, MAPE={m['MAPE']:.2f}%)",
        ASSETS / "lasso_forecast.png",
        label="Pronóstico Lasso",
    )
    print("[5] Gráfico -> assets/lasso_forecast.png")

    # 7) Tabla comparativa actualizada
    print("\n=== Comparación de modelos ===")
    rows = []
    for name in ["arima", "var", "xgboost", "lasso"]:
        try:
            row = pd.read_csv(ASSETS / f"metrics_{name}.csv").iloc[0].to_dict()
            row["Modelo"] = name.upper()
            rows.append(row)
        except FileNotFoundError:
            pass
    if rows:
        comp = pd.DataFrame(rows)[["Modelo", "MAE", "RMSE", "MAPE"]]
        comp.to_csv(ASSETS / "metrics_comparison.csv", index=False)
        print(comp.to_string(index=False))

    return m


def main() -> None:
    chdir_root()
    run_lasso()


if __name__ == "__main__":
    main()
