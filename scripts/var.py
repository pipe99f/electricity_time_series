"""VAR multivariado para el pico mensual de consumo (Países Bajos).

Replica la metodología del artículo: Suriya & Agusthiyar (2026), ZANCO J. Pure
Appl. Sci. 38(2). Ajusta un VAR sobre [peak_load, avg_temperature, humidity]
(diferenciadas) y evalúa el pronóstico del pico con walk-forward sobre 2024.

El módulo expone `forecast_var(history, lag, target='peak_load', h=1)` para que
el modelo híbrido lo reutilice después.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

from common import (
    chdir_root, load_gold, split_by_year, adf_test, ljung_box,
    compute_metrics, save_metrics, walk_forward, plot_forecast,
    ASSETS, TEST_YEAR,
)

COLS = ["peak_load", "avg_temperature", "humidity"]


def adf_all(data: pd.DataFrame) -> None:
    """Reporta el ADF de cada columna (informativo; el modelo usa diff)."""
    print("[2] ADF por serie (nivel):")
    for c in data.columns:
        print(f"    {c}: p={adf_test(data[c]):.4f}")


def select_lag_var(dtrain: pd.DataFrame, maxlags: int = 6) -> int:
    """Selecciona el rezago por AIC (con salvaguarda mín=1)."""
    maxlags = min(maxlags, max(1, len(dtrain) // 6))
    selector = VAR(dtrain).select_order(maxlags=maxlags)
    print(selector.summary())
    lag = max(int(selector.aic), 1)
    print(f"    Rezago elegido (AIC): {lag}")
    return lag


def forecast_var(history: pd.DataFrame, lag: int, target: str = "peak_load", h: int = 1):
    """Primitiva pronosticadora del pico (reutilizable por el híbrido).

    Diferencia el historial, ajusta un VAR(lag) y pronostica 1 paso; des-diferencia
    el pico como nivel_anterior + diff_pronosticada. Devuelve (pred, None, None).
    """
    hdiff = history.diff().dropna()
    f = VAR(hdiff).fit(lag).forecast(hdiff.values, steps=h)
    last_level = float(history[target].iloc[-1])
    return last_level + float(f[0, 0]), None, None


def run_var() -> dict:
    """Orquesta el pipeline completo de VAR. Devuelve el dict de métricas."""
    # 1) Series
    data = load_gold(cols=COLS).astype(float)
    print(f"[1] Series: {COLS} | {len(data)} obs")

    # 2) ADF por serie
    adf_all(data)

    train, test = split_by_year(data)

    # 3) Selección de rezago sobre series diferenciadas
    dtrain = train.diff().dropna()
    print("[3] Selección de rezago (series diferenciadas):")
    lag = select_lag_var(dtrain)

    # 4) Ajuste + Ljung-Box sobre residuos del pico
    res = VAR(dtrain).fit(lag)
    print(f"[4] VAR({lag}) ajustado. AIC={res.aic:.1f}")
    print(f"    Ljung-Box residuos peak_load p={ljung_box(np.asarray(res.resid)[:, 0], 10):.4f}")

    # 5) Walk-forward h=1 sobre 2024
    print(f"[5] Walk-forward h=1 sobre {TEST_YEAR} ({len(test)} meses) ...")
    preds, _, _ = walk_forward(
        lambda hist: forecast_var(hist, lag=lag, target="peak_load", h=1),
        train, test,
    )

    # 6) Métricas
    m = compute_metrics(test["peak_load"], preds)
    print(f"[6] MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  MAPE={m['MAPE']:.2f}%")
    save_metrics("var", m)

    # 7) Gráfico
    plot_forecast(data["peak_load"], preds, test.index, None, None,
                  "VAR - Pronóstico del pico mensual (Países Bajos)",
                  ASSETS / "var_forecast.png", label=f"VAR (MAPE={m['MAPE']:.2f}%)")
    print("[7] Gráfico -> assets/var_forecast.png")

    # 8) Tabla comparativa ARIMA vs VAR
    try:
        m_arima = pd.read_csv(ASSETS / "metrics_arima.csv").iloc[0]
        comp = pd.DataFrame({
            "Modelo": ["ARIMA", "VAR"],
            "MAE": [m_arima["MAE"], m["MAE"]],
            "RMSE": [m_arima["RMSE"], m["RMSE"]],
            "MAPE_%": [m_arima["MAPE"], m["MAPE"]],
        })
        comp.to_csv(ASSETS / "metrics_comparison.csv", index=False)
        print("\n=== Comparación ARIMA vs VAR ===")
        print(comp.to_string(index=False))
    except FileNotFoundError:
        print("(Corre primero scripts/arima.py para generar la comparación.)")
    return m


def main() -> None:
    chdir_root()
    run_var()


if __name__ == "__main__":
    main()