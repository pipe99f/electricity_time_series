"""Modelo Ridge para el pico mensual de consumo de Países Bajos.

Incluye ingeniería de variables, selección de lambda y validación
walk-forward sobre 2024.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import (
    chdir_root,
    load_gold,
    split_by_year,
    compute_metrics,
    save_metrics,
    ASSETS,
    TEST_YEAR,
)


FEATURES = [
    "avg_temperature",
    "humidity",
    "urban_population",
    "year",
    "mes_seno",
    "mes_coseno",
    "lag_1",
    "lag_3",
    "lag_6",
    "rolling_3",
    "rolling_6",
    "temp_humedad",
    "pct_change",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea las variables utilizadas por Ridge."""

    df = df.copy().interpolate().ffill()

    df["year"] = df.index.year
    df["mes_seno"] = np.sin(2 * np.pi * df.index.month / 12)
    df["mes_coseno"] = np.cos(2 * np.pi * df.index.month / 12)

    df["lag_1"] = df["peak_load"].shift(1)
    df["lag_2"] = df["peak_load"].shift(2)
    df["lag_3"] = df["peak_load"].shift(3)
    df["lag_6"] = df["peak_load"].shift(6)

    # Promedios móviles usando solamente datos anteriores
    past_peak = df["peak_load"].shift(1)

    df["rolling_3"] = past_peak.rolling(3).mean()
    df["rolling_6"] = past_peak.rolling(6).mean()

    df["temp_humedad"] = (
        df["avg_temperature"] * df["humidity"]
    )

    df["pct_change"] = (
        (df["lag_1"] / df["lag_2"]) - 1
    ) * 100

    return df.dropna()


def select_lambda(train_df: pd.DataFrame) -> tuple[float, float]:
    """Selecciona lambda mediante validación cruzada temporal."""

    model_cv = Pipeline([
        ("scaler", StandardScaler()),
        (
            "ridge",
            RidgeCV(
                alphas=np.logspace(-3, 5, 300),
                cv=TimeSeriesSplit(n_splits=5),
                scoring="neg_root_mean_squared_error",
            ),
        ),
    ])

    model_cv.fit(
        train_df[FEATURES],
        train_df["peak_load"],
    )

    ridge_cv = model_cv.named_steps["ridge"]

    return float(ridge_cv.alpha_), float(-ridge_cv.best_score_)


def walk_forward_ridge(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    alpha: float,
) -> pd.Series:
    """Pronostica 2024 reestimando Ridge cada mes."""

    history = train_df.copy()
    predictions = []

    for date in test_df.index:

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ])

        model.fit(
            history[FEATURES],
            history["peak_load"],
        )

        prediction = model.predict(
            test_df.loc[[date], FEATURES]
        )[0]

        predictions.append(float(prediction))

        # Se agrega el mes real después de realizar la predicción
        history = pd.concat([
            history,
            test_df.loc[[date]],
        ])

    return pd.Series(
        predictions,
        index=test_df.index,
        name="predicted_ridge",
    )


def run_ridge() -> dict:
    """Ejecuta el pipeline completo del modelo Ridge."""

    # 1. Cargar datos
    df = load_gold(cols=[
        "peak_load",
        "avg_temperature",
        "humidity",
        "urban_population",
    ])

    print(
        f"[1] Dataset: {len(df)} observaciones "
        f"({df.index[0].date()} -> {df.index[-1].date()})"
    )

    # 2. Crear variables
    df_features = build_features(df)

    print(
        f"[2] Variables creadas: {len(FEATURES)} | "
        f"Observaciones útiles: {len(df_features)}"
    )

    # 3. Separar entrenamiento y prueba
    train_df, test_df = split_by_year(df_features)

    print(
        f"[3] Train={len(train_df)} | "
        f"Test={len(test_df)}"
    )

    # 4. Seleccionar lambda
    alpha, cv_rmse = select_lambda(train_df)

    print(
        f"[4] Lambda óptimo={alpha:.4f} | "
        f"CV-RMSE={cv_rmse:.2f}"
    )

    # 5. Validación walk-forward
    predictions = walk_forward_ridge(
        train_df,
        test_df,
        alpha,
    )

    # 6. Métricas
    metrics = compute_metrics(
        test_df["peak_load"],
        predictions,
    )

    save_metrics("ridge", metrics)

    print("\n--- RESUMEN RIDGE ---")
    print(
        f"MAE={metrics['MAE']:.2f} | "
        f"RMSE={metrics['RMSE']:.2f} | "
        f"MAPE={metrics['MAPE']:.2f}%"
    )

    # 7. Guardar resultados
    comparison = pd.DataFrame({
        "real": test_df["peak_load"],
        "predicted_ridge": predictions,
    })

    comparison["error_absoluto"] = (
        comparison["real"]
        - comparison["predicted_ridge"]
    ).abs()

    comparison["error_porcentual"] = (
        comparison["error_absoluto"]
        / comparison["real"]
    ) * 100

    comparison.to_csv(
        ASSETS / "ridge_comparison.csv"
    )

    # 8. Gráfico
    plt.figure(figsize=(9, 5))

    plt.plot(
        test_df.index,
        test_df["peak_load"],
        marker="o",
        label="Carga real",
    )

    plt.plot(
        predictions.index,
        predictions,
        marker="o",
        label="Predicción Ridge",
    )

    plt.xlabel("Mes")
    plt.ylabel("Carga máxima")
    plt.title(
        f"Ridge - Pronóstico del pico mensual "
        f"(Países Bajos, {TEST_YEAR})"
    )

    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        ASSETS / "ridge_real_vs_predicted.png",
        dpi=300,
    )

    plt.close()

    print("[5] Resultados guardados en assets/")

    return metrics


def main() -> None:
    chdir_root()
    run_ridge()


if __name__ == "__main__":
    main()
