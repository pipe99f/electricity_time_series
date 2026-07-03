"""XGBoost para el pico mensual de consumo (Países Bajos).

Modelo de aprendizaje automático basado en características (lags, medias móviles,
variables cíclicas, exógenas climáticas) evaluado con validación walk-forward
sobre 2024, siguiendo la misma metodología que ARIMA y VAR.

Referencia: Suriya & Agusthiyar (2026), ZANCO J. Pure Appl. Sci. 38(2).
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler

from common import (
    chdir_root, load_gold, split_by_year,
    compute_metrics, save_metrics, plot_forecast,
    ASSETS, TEST_YEAR,
)

COLS = ["peak_load", "avg_temperature", "humidity", "urban_population"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ingeniería de características alineada al artículo.

    Genera: lags (1, 3, 6), media móvil (3), codificación cíclica del mes,
    interacción temperatura × humedad.
    """
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


def run_xgboost() -> dict:
    """Pipeline completo de XGBoost con walk-forward."""

    # 1) Carga y feature engineering
    df = load_gold(cols=COLS).astype(float)
    df_feat = build_features(df)
    print(f"[1] Dataset con features: {len(df_feat)} obs, {df_feat.shape[1]} columnas")

    # 2) Definir features (todo menos el target)
    feature_cols = [c for c in df_feat.columns if c != "peak_load"]

    # 3) Split temporal
    train_df, test_df = split_by_year(df_feat)
    print(f"[2] Split: train={len(train_df)} obs, test={len(test_df)} obs")

    # 4) Walk-forward h=1 (scaler ajustado SOLO en train de cada paso)
    print(f"[3] Walk-forward h=1 sobre {TEST_YEAR} ({len(test_df)} meses)...")
    history_df = train_df.copy()
    preds = []

    for i in range(len(test_df)):
        # Normalizar solo con datos de entrenamiento del paso actual
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(history_df[feature_cols])
        y_train = history_df["peak_load"].values

        X_test_row = scaler.transform(test_df[feature_cols].iloc[[i]])

        # Entrenar XGBoost (hiperparámetros conservadores para muestra pequeña)
        model = XGBRegressor(
            n_estimators=80,
            learning_rate=0.05,
            max_depth=3,
            min_child_weight=3,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=1.0,
            reg_lambda=2.0,
            random_state=42,
            verbosity=0,
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
    save_metrics("xgboost", m)

    # 6) Gráfico
    plot_forecast(
        df_feat["peak_load"], preds_series, test_df.index, None, None,
        f"XGBoost - Pronóstico del pico mensual (Países Bajos, MAPE={m['MAPE']:.2f}%)",
        ASSETS / "xgboost_forecast.png",
        label="Pronóstico XGBoost",
    )
    print("[5] Gráfico -> assets/xgboost_forecast.png")

    # 7) Tabla comparativa actualizada
    print("\n=== Comparación de modelos ===")
    rows = []
    for name in ["arima", "var", "xgboost"]:
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
    run_xgboost()


if __name__ == "__main__":
    main()
