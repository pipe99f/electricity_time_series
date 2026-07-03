from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Rutas
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "gold_netherlands.csv"
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)


# Variables del modelo
FEATURES = [
    "avg_temperature",
    "humidity",
    "urban_population",
    "year",
    "month_sin",
    "month_cos",
    "lag_1",
    "lag_3",
    "lag_6",
    "rolling_3",
    "rolling_6",
    "temp_humidity",
    "pct_change_lagged",
]


def mape(y_true, y_pred):
    """Error porcentual absoluto medio."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean(
        np.abs((y_true - y_pred) / y_true)
    ) * 100


# ============================================================
# 1. Cargar y limpiar los datos
# ============================================================

df = pd.read_csv(DATA)

df["month"] = pd.to_datetime(df["month"])

numeric_columns = [
    "peak_load",
    "avg_temperature",
    "humidity",
    "urban_population",
]

df[numeric_columns] = df[numeric_columns].apply(
    pd.to_numeric,
    errors="coerce",
)

df = (
    df.sort_values("month")
    .drop_duplicates("month")
    .dropna()
    .reset_index(drop=True)
)


# ============================================================
# 2. Ingeniería de variables
# ============================================================

df["year"] = df["month"].dt.year
df["month_num"] = df["month"].dt.month

df["month_sin"] = np.sin(
    2 * np.pi * df["month_num"] / 12
)

df["month_cos"] = np.cos(
    2 * np.pi * df["month_num"] / 12
)

df["lag_1"] = df["peak_load"].shift(1)
df["lag_2"] = df["peak_load"].shift(2)
df["lag_3"] = df["peak_load"].shift(3)
df["lag_6"] = df["peak_load"].shift(6)

# Solo se usa información pasada
past_peak = df["peak_load"].shift(1)

df["rolling_3"] = past_peak.rolling(3).mean()
df["rolling_6"] = past_peak.rolling(6).mean()

df["temp_humidity"] = (
    df["avg_temperature"] * df["humidity"]
)

df["pct_change_lagged"] = (
    (df["lag_1"] / df["lag_2"]) - 1
) * 100

df = df.dropna().reset_index(drop=True)


# ============================================================
# 3. Entrenamiento y prueba
# ============================================================

train = df[df["month"] < "2024-01-01"]
test = df[df["month"] >= "2024-01-01"]

X_train = train[FEATURES]
y_train = train["peak_load"]

X_test = test[FEATURES]
y_test = test["peak_load"]


# ============================================================
# 4. Ajuste del modelo Ridge
# ============================================================

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge()),
])

search = GridSearchCV(
    estimator=pipeline,
    param_grid={
        "ridge__alpha": np.logspace(-3, 5, 300)
    },
    scoring="neg_root_mean_squared_error",
    cv=TimeSeriesSplit(n_splits=5),
    n_jobs=-1,
)

search.fit(X_train, y_train)

model = search.best_estimator_
predictions = model.predict(X_test)

best_lambda = search.best_params_["ridge__alpha"]
cv_rmse = -search.best_score_


# ============================================================
# 5. Métricas
# ============================================================

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

mape_value = mape(y_test, predictions)

metrics = pd.DataFrame({
    "Modelo": ["Ridge"],
    "Lambda": [best_lambda],
    "CV_RMSE": [cv_rmse],
    "MAE": [mae],
    "RMSE": [rmse],
    "MAPE": [mape_value],
})

metrics.to_csv(
    ASSETS / "metrics_ridge.csv",
    index=False,
)


# ============================================================
# 6. Valores reales y predichos
# ============================================================

comparison = pd.DataFrame({
    "month": test["month"].to_numpy(),
    "real": y_test.to_numpy(),
    "predicted_ridge": predictions,
})

comparison["absolute_error"] = np.abs(
    comparison["real"]
    - comparison["predicted_ridge"]
)

comparison["percentage_error"] = (
    comparison["absolute_error"]
    / comparison["real"]
) * 100

comparison.to_csv(
    ASSETS / "ridge_comparison.csv",
    index=False,
)


# ============================================================
# 7. Gráfico
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    comparison["month"],
    comparison["real"],
    marker="o",
    label="Carga real",
)

plt.plot(
    comparison["month"],
    comparison["predicted_ridge"],
    marker="o",
    label="Predicción Ridge",
)

plt.xlabel("Mes")
plt.ylabel("Carga máxima")
plt.title("Carga máxima real y predicha con Ridge")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

plt.savefig(
    ASSETS / "ridge_real_vs_predicted.png",
    dpi=300,
)

plt.close()


# ============================================================
# 8. Mostrar resultados
# ============================================================

print("\nModelo Ridge ejecutado correctamente")
print(f"Lambda óptimo: {best_lambda:.3f}")
print(f"CV-RMSE: {cv_rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape_value:.2f}%")

print("\nPredicciones:")
print(comparison)
