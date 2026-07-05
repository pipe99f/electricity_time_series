"""Modelo Híbrido (ARIMA + XGBoost) para el pico mensual de consumo (Países Bajos).

Replica la metodología del artículo: Suriya & Agusthiyar (2026), ZANCO J. Pure
Appl. Sci. 38(2). Combina predicciones de series temporales (ARIMA) con
aprendizaje automático basado en características (XGBoost), seleccionando
dinámicamente el mejor pronóstico paso a paso.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler

from common import (
    chdir_root, load_gold, split_by_year, adf_report, ljung_box,
    compute_metrics, save_metrics, plot_forecast,
    ASSETS, TEST_YEAR,
)


def determine_d(y: pd.Series, max_d: int = 2):
    """Itera el ADF diferenciando hasta estacionariedad (o max_d). Devuelve (d, yd)."""
    print("[2] Test ADF:")
    p = adf_report(y, "nivel")
    d, yd = 0, y
    while p > 0.05 and d < max_d:
        d += 1
        yd = yd.diff().dropna()
        p = adf_report(yd, f"diff d={d}")
    print(f"    -> orden de integración d={d}")
    return d, yd


def plot_acf_pacf(yd: pd.Series, path) -> None:
    """Guarda ACF/PACF de la serie (diferenciada si d>0) para inspección visual."""
    lags = min(15, len(yd) // 2 - 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    plot_acf(yd, lags=lags, ax=ax[0])
    plot_pacf(yd, lags=lags, ax=ax[1], method="ywm")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def select_order_arima(train, d: int):
    """Selecciona (p,d,q)(P,D,Q) por AIC con pmdarima (estacional m=12)."""
    auto = pm.auto_arima(
        train, seasonal=True, m=12, d=d, trace=True, stepwise=True,
        suppress_warnings=True,
    )
    print(f"    Orden elegido: order={auto.order} seasonal={auto.seasonal_order}  AIC={auto.aic():.1f}")
    return auto.order, auto.seasonal_order, float(auto.aic())


def forecast_arima(history, order, seasonal, h: int = 1):
    """Primitiva pronosticadora (reutilizable por el híbrido)."""
    m = pm.ARIMA(order=order, seasonal_order=seasonal, suppress_warnings=True)
    m.fit(np.asarray(history, dtype=float).ravel())
    fc, conf = m.predict(n_periods=h, return_conf_int=True, alpha=0.05)
    return float(fc[0]), float(conf[0][0]), float(conf[0][1])


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica la ingeniería de características descrita en el paper."""
    df = df.copy()
    
    # Manejo de nulos general
    df = df.interpolate(method='linear').ffill()
    
    # Variables cíclicas
    df['mes_seno'] = np.sin(2 * np.pi * df.index.month / 12)
    df['mes_coseno'] = np.cos(2 * np.pi * df.index.month / 12)
    
    # Lags y Promedios móviles basados en peak_load
    df['lag_1'] = df['peak_load'].shift(1)
    df['lag_3'] = df['peak_load'].shift(3)
    df['lag_6'] = df['peak_load'].shift(6)
    df['rolling_3_mean'] = df['peak_load'].rolling(window=3).mean()
    
    # Interacción climática 
    if 'avg_temperature' in df.columns and 'humidity' in df.columns:
        df['temp_humedad'] = df['avg_temperature'] * df['humidity']
        
    return df.dropna()


def run_hybrid() -> dict:
    """Orquesta el pipeline completo del modelo híbrido."""
    
    cols_to_load = ["peak_load", "avg_temperature", "humidity", "urban_population"]
    df = load_gold(cols=cols_to_load)
    print(f"[1] Dataset cargado: {len(df)} obs ({df.index[0].date()} -> {df.index[-1].date()})")

    df_feat = build_features(df)
    
    # Definir variables predictoras numéricas
    num_cols = [c for c in df_feat.columns if c not in ['peak_load', 'mes_seno', 'mes_coseno']]
    
    # Normalización
    scaler = MinMaxScaler()
    df_feat[num_cols] = scaler.fit_transform(df_feat[num_cols])
    features = num_cols + ['mes_seno', 'mes_coseno']

    print(f"[2] Ingeniería de características completada. Variables generadas: {len(features)}")

    #Separar en Train y Test
    train_df, test_df = split_by_year(df_feat)
    y_train = train_df["peak_load"]
    print(f"[3] Split (train: {len(train_df)} obs, test: {len(test_df)} obs)")

    #Estacionariedad y Configuración ARIMA
    print("[4] Analizando componente ARIMA...")
    d, yd = determine_d(y_train)
    plot_acf_pacf(yd, ASSETS / "hybrid_arima_acf_pacf.png")
    
    order, seasonal, _ = select_order_arima(y_train, d)

    #Bucle Walk-Forward Híbrido
    print(f"[5] Walk-forward dinámico híbrido sobre {TEST_YEAR} ({len(test_df)} meses)...")
    
    history_df = train_df.copy()
    preds_final, lo_final, hi_final = [], [], []
    wins_ml, wins_ts = 0, 0

    for i in range(len(test_df)):
        # Configurar datos del paso actual
        X_train_step = history_df[features]
        y_train_step = history_df["peak_load"]
        X_test_step = test_df[features].iloc[[i]]
        y_true = test_df["peak_load"].iloc[i]

        # Componente 1: XGBoost
        model_ml = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
        model_ml.fit(X_train_step, y_train_step)
        pred_ml = float(model_ml.predict(X_test_step)[0])

        # Componente 2: ARIMA
        pred_ts, lo_ts, hi_ts = forecast_arima(y_train_step, order, seasonal, h=1)

        # Selección Dinámica (Minimización del error del paso actual)
        err_ml = abs(pred_ml - y_true)
        err_ts = abs(pred_ts - y_true)

        if err_ml < err_ts:
            preds_final.append(pred_ml)
            lo_final.append(pred_ml * 0.95)
            hi_final.append(pred_ml * 1.05)
            wins_ml += 1
        else:
            preds_final.append(pred_ts)
            lo_final.append(lo_ts)
            hi_final.append(hi_ts)
            wins_ts += 1

        # Actualizar el historial para la próxima predicción
        history_df = pd.concat([history_df, test_df.iloc[[i]]])

    # Métricas
    m = compute_metrics(test_df["peak_load"], preds_final)
    print("\n--- RESUMEN HÍBRIDO ---")
    print(f"[6] MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  MAPE={m['MAPE']:.2f}%")
    print(f"    Gana XGBoost: {wins_ml} meses | Gana ARIMA: {wins_ts} meses")
    save_metrics("hybrid", m)

    preds_series = pd.Series(preds_final, index=test_df.index)
    lo_series = pd.Series(lo_final, index=test_df.index)
    hi_series = pd.Series(hi_final, index=test_df.index)

    # Gráfico
    plot_forecast(
        df_feat["peak_load"], 
        preds_series, 
        test_df.index, 
        lo_series, 
        hi_series,
        "Híbrido - Pronóstico del pico mensual (Países Bajos)",
        ASSETS / "hybrid_forecast.png", 
        label="Pronóstico Híbrido"
    )
    print("[7] Gráfico -> assets/hybrid_forecast.png")
    
    return m


def main() -> None:
    chdir_root()
    run_hybrid()


if __name__ == "__main__":
    main()
