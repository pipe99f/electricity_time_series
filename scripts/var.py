"""
VAR multivariado para el pico mensual de consumo electrico (Netherlands)
========================================================================

Ajusta un modelo VAR (Vector Autoregression) sobre el conjunto de series
[peak_load, avg_temperature, humidity] del dataset gold y evalua el pronostico
del pico con validacion walk-forward, replicando la metodologia del articulo:
  Suriya & Agusthiyar (2026), ZANCO J. Pure Appl. Sci. 38(2).

El VAR, a diferencia del ARIMA (univariado), captura interdependencias entre
varias series (aqui: consumo, temperatura y humedad), que es justamente el uso
que el paper le da al VAR.

Pipeline:
  1. Cargar data/gold_netherlands.csv; series = [peak_load, avg_temperature, humidity].
  2. Test ADF por serie (informativo). Para el modelo se diferencian TODAS las
     series (simplificacion que garantiza estacionariedad conjunta).
  3. Seleccion de rezago optimo por AIC/BIC/HQIC con statsmodels.VAR.
  4. Ajuste VAR y diagnostico de residuos (Ljung-Box sobre peak_load).
  5. Walk-forward h=1 sobre 2024: se reestima el VAR cada paso y se pronostica
     el pico (des-diferenciando: nivel anterior + pronostico de la diff).
  6. Metricas MAE, RMSE, MAPE para peak_load.
  7. Grafico pronostico vs real -> assets/var_forecast.png y tabla comparativa
     ARIMA vs VAR -> assets/metrics_comparison.csv.

Uso:
    python scripts/var.py
Dependencias: pandas, numpy, matplotlib, statsmodels
"""

from pathlib import Path
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from statsmodels.stats.diagnostic import acorr_ljungbox

GOLD_PATH = Path("data/gold_netherlands.csv")
ASSETS = Path("assets")
TEST_YEAR = 2024

COLS = ["peak_load", "avg_temperature", "humidity"]


def adf_p(s: pd.Series) -> float:
    """p-valor del test ADF (H0: raiz unitaria). p<0.05 => estacionaria."""
    return float(adfuller(s.dropna())[1])


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    ASSETS.mkdir(exist_ok=True)

    # 1) Cargar las series multivariadas
    df = pd.read_csv(GOLD_PATH, parse_dates=["month"], index_col="month")
    data = df[COLS].astype(float)
    print(f"[1] Series: {COLS} | {len(data)} obs")

    # 2) ADF por serie (informativo). Modelaremos sobre las series diferenciadas.
    print("[2] ADF por serie (nivel):")
    for c in COLS:
        print(f"    {c}: p={adf_p(data[c]):.4f}")

    # Particion train/test (a nivel de la serie original, para des-diferenciar)
    train = data[data.index.year < TEST_YEAR]
    test = data[data.index.year == TEST_YEAR]

    # 3) Seleccion de rezago sobre las series DIFERENCIADAS (estacionarias)
    dtrain = train.diff().dropna()
    # maxlags acotado por la cantidad de datos (serie corta: 47 obs)
    maxlags = min(6, max(1, len(dtrain) // 6))
    print(f"[3] Seleccion de rezago (maxlags={maxlags}, series diferenciadas):")
    selector = VAR(dtrain).select_order(maxlags=maxlags)
    print(selector.summary())
    lag = int(selector.aic)                 # rezago elegido por AIC
    # salvaguarda: si AIC elige 0, usamos 1 (un VAR(0) no captura dinamica)
    lag = max(lag, 1)
    print(f"    Rezago elegido (AIC): {lag}")

    # 4) Ajuste VAR y diagnostico de residuos de peak_load
    res = VAR(dtrain).fit(lag)
    print(f"[4] VAR({lag}) ajustado. AIC={res.aic:.1f}")
    resid_arr = np.asarray(res.resid)             # (n_obs, k) -> array por compatibilidad
    lb = acorr_ljungbox(resid_arr[:, 0], lags=[min(10, len(resid_arr) - 1)],
                        return_df=True)
    print(f"    Ljung-Box residuos peak_load p={float(lb['lb_pvalue'].iloc[0]):.4f}")

    # 5) Walk-forward h=1 sobre 2024. En cada paso:
    #    - se diferencia el historial, se reestima el VAR(lag) y se pronostica 1 paso
    #    - se des-diferencia el pico: y_t_hat = y_{t-1} (nivel) + diff_hat
    print(f"[5] Walk-forward h=1 sobre {TEST_YEAR} ({len(test)} meses) ...")
    history = train.copy()
    preds = []
    for t in range(len(test)):
        hdiff = history.diff().dropna()
        m = VAR(hdiff).fit(lag)
        f = m.forecast(hdiff.values, steps=1)          # pronostico en diferencias
        last_level = float(history["peak_load"].iloc[-1])
        pred_peak = last_level + float(f[0, 0])        # des-diferenciar el pico
        preds.append(pred_peak)
        history = pd.concat([history, test.iloc[[t]]])
    preds = pd.Series(preds, index=test.index, name="peak_load_pred")

    # 6) Metricas para peak_load
    actual = test["peak_load"]
    mae = float(np.mean(np.abs(actual - preds)))
    rmse = float(np.sqrt(np.mean((actual - preds) ** 2)))
    mp = float(mape(actual, preds))
    print(f"[6] MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mp:.2f}%")
    pd.DataFrame({"MAE": [mae], "RMSE": [rmse], "MAPE": [mp]}).to_csv(
        ASSETS / "metrics_var.csv", index=False)

    # 7) Grafico del pronostico VAR
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(data.index, data["peak_load"], label="Real", lw=1.6)
    ax.plot(preds.index, preds, "--", label=f"VAR (MAPE={mp:.2f}%)", lw=1.6)
    ax.axvspan(test.index[0], test.index[-1], alpha=0.08, color="gray",
               label=f"Test {TEST_YEAR}")
    ax.set_title("VAR - Pronostico del pico mensual (Netherlands)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ASSETS / "var_forecast.png", dpi=130)
    plt.close(fig)
    print("[7] Grafico -> assets/var_forecast.png")

    # 8) Tabla comparativa ARIMA vs VAR (lee las metricas de ARIMA)
    try:
        m_arima = pd.read_csv(ASSETS / "metrics_arima.csv").iloc[0]
        comp = pd.DataFrame({
            "Modelo": ["ARIMA", "VAR"],
            "MAE": [m_arima["MAE"], mae],
            "RMSE": [m_arima["RMSE"], rmse],
            "MAPE_%": [m_arima["MAPE"], mp],
        })
        comp.to_csv(ASSETS / "metrics_comparison.csv", index=False)
        print("\n=== Comparacion ARIMA vs VAR ===")
        print(comp.to_string(index=False))
    except FileNotFoundError:
        print("(Corre primero scripts/arima.py para generar la comparacion.)")


if __name__ == "__main__":
    main()
