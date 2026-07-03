"""
Pipeline de limpieza de datos: raw -> gold
==========================================

Toma el dataset crudo `data/global_climate_energy_2020_2024.csv` (datos diarios
sinteticos de 20 paises, 2020-2024) y produce un dataset "gold" MENSUAL para un
solo pais, listo para modelado con series temporales.

El objetivo es replicar, sobre los datos disponibles, la metodologia del articulo:
  Suriya & Agusthiyar (2026), "Enhanced Prediction of Electricity Peak Load via
  Machine Learning and Time Series Analysis", ZANCO J. Pure Appl. Sci. 38(2).

Decisiones (alineadas al paper):
  * Pais: Netherlands (elegido como el "pais mas pequeno" del conjunto).
  * Frecuencia: MENSUAL (el paper trabaja con datos mensuales, ~60 observaciones).
  * Variable objetivo: pico mensual de consumo = maximo diario de
    `energy_consumption` en cada mes (analogo al "peak load demand" del paper).
  * Exogenas climaticas: media mensual de `avg_temperature` y `humidity`
    (el paper usa temperatura, humedad y precipitacion; aqui disponemos de
    temperatura y humedad).

Pasos:
  1. Cargar el CSV crudo.
  2. Filtrar el pais objetivo.
  3. Parsear la fecha y usarla como indice temporal.
  4. Garantizar continuidad diaria (reindex al rango completo); eliminar duplicados.
  5. Verificar e imputar NaNs (interpolacion lineal, defensivo).
  6. Winsorizacion de atipicos (percentiles 5 y 95), como en el paper.
  7. Agregacion mensual: energy_consumption -> max (pico); temp/humedad -> mean.
  8. Guardar `data/gold_netherlands.csv` y mostrar un EDA por consola.

Uso (desde la raiz del repo o desde cualquier sitio, el script se reubica solo):
    python scripts/data_cleaning.py
Salida:
    data/gold_netherlands.csv
"""

from pathlib import Path
import os

import numpy as np
import pandas as pd

# --- Configuracion central (parametros para facilitar cambios) -------------
RAW_PATH = Path("data/global_climate_energy_2020_2024.csv")
COUNTRY = "Netherlands"                                   # pais elegido
GOLD_PATH = Path(f"data/gold_{COUNTRY.lower().replace(' ', '_')}.csv")

# Variables numericas a winsorizar (excluyen fecha y pais, que no son metricas)
NUMERIC_COLS = [
    "avg_temperature", "humidity", "co2_emission", "energy_consumption",
    "renewable_share", "urban_population", "industrial_activity_index",
    "energy_price",
]


def winsorize(s: pd.Series, lo: float = 0.05, hi: float = 0.95) -> pd.Series:
    """Acota los valores extremos a los percentiles `lo` y `hi`.

    Reproduce la Winsorizacion del paper (P5/P95): en lugar de ELIMINAR los
    atipicos, los RECORTA para mitigar su influencia conservando la variabilidad
    estacional. Es lo que hace el articulo con los picos de demanda atipicos.
    """
    q_lo, q_hi = s.quantile(lo), s.quantile(hi)
    return s.clip(lower=q_lo, upper=q_hi)


def main() -> None:
    # Garantiza que las rutas relativas funcionen desde cualquier cwd.
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)

    # 1) Carga del dataset crudo
    df = pd.read_csv(RAW_PATH)
    print(f"[1] Cargado {RAW_PATH}: {df.shape[0]} filas, {df.shape[1]} columnas")

    # 2) Filtrado del pais
    df = df[df["country"] == COUNTRY].copy()
    print(f"[2] Filtrado '{COUNTRY}': {df.shape[0]} filas diarias")

    # 3) Fecha -> indice temporal ordenado; quitamos la columna country (constante)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df.drop(columns=["country"])

    # 4) Continuidad diaria: duplicados y rango completo
    n_dup = df.index.duplicated().sum()
    if n_dup:
        df = df[~df.index.duplicated(keep="first")]
        print(f"[4] {n_dup} fechas duplicadas eliminadas")
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
    missing = full_range.difference(df.index)
    df = df.reindex(full_range)              # introduce NaN si faltaba alguna fecha
    df.index.name = "date"
    print(f"[4] Rango diario {full_range[0].date()} -> {full_range[-1].date()} "
          f"({len(full_range)} dias); fechas faltantes reindexadas: {len(missing)}")

    # 5) NaNs: verificar e imputar por interpolacion lineal (brechas cortas)
    n_nan = int(df.isna().sum().sum())
    if n_nan > 0:
        print(f"[5] {n_nan} NaNs detectados -> interpolacion lineal + ffill/bfill")
        df = df.interpolate(method="time").ffill().bfill()
    else:
        print("[5] Sin NaNs: nada que imputar")

    # 6) Agregacion MENSUAL: pico (max) del consumo; medias de las climaticas
    monthly = pd.DataFrame({
        "peak_load": df["energy_consumption"].resample("MS").max(),   # pico del mes
        "avg_temperature": df["avg_temperature"].resample("MS").mean(),
        "humidity": df["humidity"].resample("MS").mean(),
        "urban_population": df["urban_population"].resample("MS").mean(),
    })
    monthly.index.name = "month"
    print(f"[6] Agregado mensual: {len(monthly)} meses "
          f"({monthly.index[0].date()} -> {monthly.index[-1].date()})")

    # 7) Winsorizacion MENSUAL (P5/P95) sobre las series ya agregadas.
    #    Se aplica DESPUES de agregar (no en lo diario) para no crear un techo
    #    artificial en el pico mensual: se recortan solo los meses extremos,
    #    que es el espiritu del paper (mitigar picos atipicos puntuales).
    for col in monthly.columns:
        monthly[col] = winsorize(monthly[col])
    print("[7] Winsorizacion P5/P95 aplicada a las series mensuales")

    # 8) Guardar el gold y mostrar un EDA rapido
    GOLD_PATH.parent.mkdir(exist_ok=True)
    monthly.to_csv(GOLD_PATH, index=True)
    print(f"[8] Gold guardado en {GOLD_PATH}\n")
    print("--- EDA (primeras filas) ---")
    print(monthly.head())
    print("\n--- Estadisticos ---")
    print(monthly.describe().round(2).to_string())


if __name__ == "__main__":
    main()
