"""ARIMA univariado para el pico mensual de consumo (Países Bajos).

Replica la metodología del artículo: Suriya & Agusthiyar (2026), ZANCO J. Pure
Appl. Sci. 38(2). Se seleccionan automáticamente los órdenes (p,d,q)(P,D,Q)
minimizando el AIC (con estacionalidad anual) y se evalúa con validación
walk-forward sobre 2024.

El módulo expone `forecast_arima(history, order, seasonal, h=1)` para que el
modelo híbrido lo reutilice después.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm

from common import (
    chdir_root, load_gold, split_by_year, adf_report, ljung_box,
    compute_metrics, save_metrics, walk_forward, plot_forecast,
    ASSETS, TEST_YEAR,
)
from common import mape as _mape  # alias corto para uso interno


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
        error_action="ignore", suppress_warnings=True,
    )
    print(f"    Orden elegido: order={auto.order} seasonal={auto.seasonal_order}  AIC={auto.aic():.1f}")
    return auto.order, auto.seasonal_order, float(auto.aic())


def forecast_arima(history, order, seasonal, h: int = 1):
    """Primitiva pronosticadora (reutilizable por el híbrido).

    Ajusta un ARIMA con `order`/`seasonal` al `history` y devuelve (pred, lo, hi)
    para h pasos. `history` puede ser Series, array o lista.
    """
    m = pm.ARIMA(order=order, seasonal_order=seasonal,
                 suppress_warnings=True, error_action="ignore")
    m.fit(np.asarray(history, dtype=float).ravel())
    fc, conf = m.predict(n_periods=h, return_conf_int=True, alpha=0.05)
    return float(fc[0]), float(conf[0][0]), float(conf[0][1])


def run_arima() -> dict:
    """Orquesta el pipeline completo de ARIMA. Devuelve el dict de métricas."""
    # 1) Serie
    y = load_gold(cols=["peak_load"])["peak_load"].astype(float)
    print(f"[1] Serie 'peak_load': {len(y)} obs "
          f"({y.index[0].date()} -> {y.index[-1].date()})")

    # 2) Estacionariedad
    d, yd = determine_d(y)

    # 3) ACF/PACF
    plot_acf_pacf(yd, ASSETS / "arima_acf_pacf.png")
    print("[3] ACF/PACF -> assets/arima_acf_pacf.png")

    # 4) Selección de órdenes (auto_arima)
    train, test = split_by_year(y)
    print(f"[4] auto_arima (train: {len(train)} obs, test: {len(test)} obs) ...")
    order, seasonal, _ = select_order_arima(train, d)

    # 5) Ajuste + Ljung-Box
    fit = pm.ARIMA(order=order, seasonal_order=seasonal,
                   suppress_warnings=True, error_action="ignore").fit(train)
    print(f"[5] Ljung-Box(10) p={ljung_box(fit.resid(), 10):.4f} "
          "(>0.05 indica residuos sin autocorrelación)")

    # 6) Walk-forward h=1 sobre 2024
    print(f"[6] Walk-forward h=1 sobre {TEST_YEAR} ({len(test)} meses) ...")
    preds, lo, hi = walk_forward(
        lambda hist: forecast_arima(hist, order, seasonal, h=1),
        train, test,
    )

    # 7) Métricas
    m = compute_metrics(test, preds)
    print(f"[7] MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  MAPE={m['MAPE']:.2f}%")
    print(f"    IC95% (primer mes): [{lo[0]:.1f}, {hi[0]:.1f}]")
    save_metrics("arima", m)

    # 8) Gráfico
    plot_forecast(y, preds, test.index, lo, hi,
                  "ARIMA - Pronóstico del pico mensual (Países Bajos)",
                  ASSETS / "arima_forecast.png", label="Pronóstico ARIMA")
    print("[8] Gráfico -> assets/arima_forecast.png")
    return m


def main() -> None:
    chdir_root()
    run_arima()


if __name__ == "__main__":
    main()