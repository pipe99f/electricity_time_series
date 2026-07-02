"""
ARIMA univariado para el pico mensual de consumo electrico (Netherlands)
========================================================================

Ajusta un modelo ARIMA sobre la serie `peak_load` del dataset gold y lo evalua
con validacion walk-forward, replicando la metodologia del articulo:
  Suriya & Agusthiyar (2026), ZANCO J. Pure Appl. Sci. 38(2).

NOTA sobre autoArima: en Python existe el equivalente del `auto.arima` de R:
  `pmdarima.auto_arima`. Busca automaticamente los ordenes (p,d,q) -y los
  estacionales (P,D,Q)- minimizando el AIC. Se usa aqui para la seleccion
  de ordenes, igual que el paper (que usa AIC/BIC + ACF/PACF).

Pipeline:
  1. Cargar data/gold_netherlands.csv; serie = peak_load (mensual).
  2. Test ADF de estacionariedad (diferenciar mientras p > 0.05).
  3. Graficos ACF/PACF -> assets/arima_acf_pacf.png.
  4. Seleccion automatica de (p,d,q)(P,D,Q) con pmdarima.auto_arima (m=12).
  5. Ajuste del modelo y diagnostico de residuos (Ljung-Box).
  6. Validacion walk-forward sobre 2024 (h=1, re-estimando cada paso con el
     orden seleccionado).
  7. Metricas MAE, RMSE, MAPE e intervalo de confianza al 95%.
  8. Grafico pronostico vs real -> assets/arima_forecast.png; metricas a CSV.

Uso:
    python scripts/arima.py
Dependencias: pandas, numpy, matplotlib, statsmodels, pmdarima
"""

from pathlib import Path
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # backend sin GUI (guarda PNG directamente)
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import pmdarima as pm

GOLD_PATH = Path("data/gold_netherlands.csv")
ASSETS = Path("assets")
TEST_YEAR = 2024                            # walk-forward sobre los 12 meses de 2024


def adf_report(s: pd.Series, name: str) -> float:
    """Test ADF: H0 = existe raiz unitaria (serie NO estacionaria).
    Si p < 0.05 se rechaza H0 => la serie es estacionaria."""
    stat, p, *_ = adfuller(s.dropna())
    tag = "estacionaria" if p < 0.05 else "NO estacionaria"
    print(f"    ADF {name}: stat={stat:.3f}, p={p:.4f} -> {tag}")
    return p


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error en %."""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    ASSETS.mkdir(exist_ok=True)

    # 1) Cargar la serie mensual
    df = pd.read_csv(GOLD_PATH, parse_dates=["month"], index_col="month")
    y = df["peak_load"].astype(float)
    print(f"[1] Serie 'peak_load': {len(y)} obs "
          f"({y.index[0].date()} -> {y.index[-1].date()})")

    # 2) Estacionariedad: ADF en nivel y, si procede, en diferencias
    print("[2] Test ADF:")
    p = adf_report(y, "nivel")
    d = 0
    yd = y
    while p > 0.05 and d < 2:               # limitamos d a 2 (practica habitual)
        d += 1
        yd = yd.diff().dropna()
        p = adf_report(yd, f"diff d={d}")
    print(f"    -> orden de integracion d={d}")

    # 3) ACF / PACF de la serie (diferenciada si d>0) para inspeccion visual
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    plot_acf(yd, lags=min(15, len(yd) // 2 - 1), ax=ax[0])
    plot_pacf(yd, lags=min(15, len(yd) // 2 - 1), ax=ax[1], method="ywm")
    fig.tight_layout()
    fig.savefig(ASSETS / "arima_acf_pacf.png", dpi=130)
    plt.close(fig)
    print("[3] ACF/PACF -> assets/arima_acf_pacf.png")

    # 4) auto_arima: seleccion automatica de ordenes (el "autoArima" de Python)
    train = y[y.index.year < TEST_YEAR]     # 2020-2023 (entrenamiento)
    test = y[y.index.year == TEST_YEAR]     # 2024 (prueba)
    print(f"[4] auto_arima (train: {len(train)} obs, test: {len(test)} obs) ...")
    auto = pm.auto_arima(
        train, seasonal=True, m=12, d=d,
        trace=True, stepwise=True,
        error_action="ignore", suppress_warnings=True,
    )
    order, seasonal = auto.order, auto.seasonal_order
    print(f"    Orden elegido: order={order} seasonal={seasonal}  AIC={auto.aic():.1f}")

    # 5) Diagnostico de residuos: Ljung-Box (idealmente p>0.05 => sin autocorrel.)
    resid = auto.resid()
    lb = acorr_ljungbox(resid, lags=[10], return_df=True)
    print(f"[5] Ljung-Box(10) p={float(lb['lb_pvalue'].iloc[0]):.4f} "
          "(>0.05 indica residuos sin autocorrelacion)")

    # 6) Walk-forward h=1 sobre 2024: en cada paso se reestima el modelo con el
    #    orden ya seleccionado (sin volver a buscar) y se pronostica 1 mes.
    print(f"[6] Walk-forward h=1 sobre {TEST_YEAR} ({len(test)} meses) ...")
    history = list(train)
    preds, lo, hi = [], [], []
    for t in range(len(test)):
        m = pm.ARIMA(order=order, seasonal_order=seasonal,
                     suppress_warnings=True, error_action="ignore")
        m.fit(history)
        fc, conf = m.predict(n_periods=1, return_conf_int=True, alpha=0.05)
        preds.append(float(fc[0]))
        lo.append(float(conf[0][0]))
        hi.append(float(conf[0][1]))
        history.append(float(test.iloc[t]))   # se anade el real observado
    preds = pd.Series(preds, index=test.index)

    # 7) Metricas de error
    mae = float(np.mean(np.abs(test - preds)))
    rmse = float(np.sqrt(np.mean((test - preds) ** 2)))
    mp = float(mape(test, preds))
    print(f"[7] MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mp:.2f}%")
    print(f"    IC95% (primer mes): [{lo[0]:.1f}, {hi[0]:.1f}]")
    pd.DataFrame({"MAE": [mae], "RMSE": [rmse], "MAPE": [mp]}).to_csv(
        ASSETS / "metrics_arima.csv", index=False)

    # 8) Grafico pronostico vs real con banda de IC 95%
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(y.index, y, label="Real", lw=1.6)
    ax.plot(preds.index, preds, "--", label="Pronostico ARIMA", lw=1.6)
    ax.fill_between(preds.index, lo, hi, alpha=0.2, label="IC 95%")
    ax.axvspan(test.index[0], test.index[-1], alpha=0.08, color="gray",
               label=f"Test {TEST_YEAR}")
    ax.set_title("ARIMA - Pronostico del pico mensual (Netherlands)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ASSETS / "arima_forecast.png", dpi=130)
    plt.close(fig)
    print("[8] Grafico -> assets/arima_forecast.png")


if __name__ == "__main__":
    main()
