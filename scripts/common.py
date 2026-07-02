"""Utilidades compartidas para los scripts de modelado (ARIMA, VAR, híbrido).

Centraliza carga de datos, tests estadísticos, métricas, validación walk-forward
y graficado, para evitar duplicación y permitir que el modelo híbrido reutilice
el mismo pipeline que ARIMA y VAR (como en el artículo de Suriya & Agusthiyar,
2026)."""

from pathlib import Path
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

GOLD_PATH = Path("data/gold_netherlands.csv")
ASSETS = Path("assets")
TEST_YEAR = 2024


def chdir_root() -> Path:
    """Se ubica en la raíz del repo (independientemente del cwd de lanzamiento)."""
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    ASSETS.mkdir(exist_ok=True)
    return root


def load_gold(cols=None) -> pd.DataFrame:
    """Carga el dataset gold mensual. `cols=None` -> todas las columnas."""
    df = pd.read_csv(GOLD_PATH, parse_dates=["month"], index_col="month")
    return df if cols is None else df[cols]


def split_by_year(df: pd.DataFrame, test_year: int = TEST_YEAR):
    """Parte el DataFrame en train (< test_year) y test (== test_year)."""
    return df[df.index.year < test_year], df[df.index.year == test_year]


def adf_test(s: pd.Series) -> float:
    """p-valor del test ADF (H0: raiz unitaria). p<0.05 => estacionaria."""
    return float(adfuller(s.dropna())[1])


def adf_report(s: pd.Series, name: str) -> float:
    """ADF con impresión legible. Devuelve el p-valor."""
    stat, p = adfuller(s.dropna())[:2]
    tag = "estacionaria" if p < 0.05 else "NO estacionaria"
    print(f"    ADF {name}: stat={stat:.3f}, p={p:.4f} -> {tag}")
    return p


def ljung_box(resid, lags: int = 10) -> float:
    """p-valor del test de Ljung-Box sobre los residuos (p>0.05 => sin autocorr.)."""
    r = np.asarray(resid).ravel()
    lb = acorr_ljungbox(r, lags=[min(lags, len(r) - 1)], return_df=True)
    return float(lb["lb_pvalue"].iloc[0])


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mape(y_true, y_pred) -> float:
    return float(np.mean(np.abs((np.asarray(y_true) - np.asarray(y_pred)) / np.asarray(y_true))) * 100)


def compute_metrics(actual, pred) -> dict:
    """Devuelve {'MAE','RMSE','MAPE'} como floats."""
    return {"MAE": mae(actual, pred), "RMSE": rmse(actual, pred), "MAPE": mape(actual, pred)}


def save_metrics(name: str, metrics: dict) -> None:
    """Guarda assets/metrics_<name>.csv con una fila de métricas."""
    pd.DataFrame([metrics]).to_csv(ASSETS / f"metrics_{name}.csv", index=False)


def walk_forward(forecast_fn, train, test, **kw):
    """Validación walk-forward h=1.

    `forecast_fn(history, **kw) -> (pred, lo, hi)` se llama en cada paso con el
    historial acumulado; `lo`/`hi` pueden ser None si el modelo no da IC.
    Devuelve (preds: Series, lo: list|None, hi: list|None).
    """
    history = train.copy()
    preds, lo, hi = [], [], []
    for t in range(len(test)):
        pred, l, h = forecast_fn(history, **kw)
        preds.append(float(pred))
        lo.append(None if l is None else float(l))
        hi.append(None if h is None else float(h))
        history = pd.concat([history, test.iloc[[t]]])
    preds = pd.Series(preds, index=test.index, name="pred")
    return preds, (None if all(x is None for x in lo) else lo), (None if all(x is None for x in hi) else hi)


def plot_forecast(full, preds, test_idx, lo, hi, title, path, label="Pronóstico"):
    """Guarda un PNG del pronóstico vs real con banda IC opcional y zona de test."""
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(full.index, full, label="Real", lw=1.6)
    ax.plot(preds.index, preds, "--", label=label, lw=1.6)
    if lo is not None and hi is not None:
        ax.fill_between(preds.index, lo, hi, alpha=0.2, label="IC 95%")
    ax.axvspan(test_idx[0], test_idx[-1], alpha=0.08, color="gray", label=f"Test {test_idx[0].year}")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)