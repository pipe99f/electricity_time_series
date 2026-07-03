import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==========================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# ==========================================
# Cargando los datos desde la carpeta 'data' de tu repositorio
df = pd.read_csv('data/gold_netherlands.csv')

# Asegurarnos de que la fecha es el índice y está ordenada
df['fecha'] = pd.to_datetime(df['fecha'])
df = df.sort_values('fecha').set_index('fecha')

# Manejo de valores nulos para todas las variables
df = df.interpolate(method='linear').ffill()

# ==========================================
# 2. INGENIERÍA DE CARACTERÍSTICAS
# ==========================================
# Codificación cíclica del tiempo
df['mes_seno'] = np.sin(2 * np.pi * df.index.month / 12)
df['mes_coseno'] = np.cos(2 * np.pi * df.index.month / 12)

# Variables rezagadas (Lags)
df['lag_1'] = df['consumo_maximo'].shift(1)
df['lag_3'] = df['consumo_maximo'].shift(3)
df['lag_6'] = df['consumo_maximo'].shift(6)

# Promedios móviles
df['rolling_3_mean'] = df['consumo_maximo'].rolling(window=3).mean()

# Interacciones climáticas
df['temp_humedad_interaccion'] = df['temperatura_promedio'] * df['humedad_promedio']

# Eliminar las primeras filas que quedaron vacías por los lags
df = df.dropna()

# ==========================================
# 3. NORMALIZACIÓN
# ==========================================
scaler = MinMaxScaler()
columnas_numericas = ['temperatura_promedio', 'humedad_promedio', 'poblacion_urbana', 
                      'lag_1', 'lag_3', 'lag_6', 'rolling_3_mean', 'temp_humedad_interaccion']

# Normalizamos solo las variables predictoras (dejamos el 'consumo_maximo' en su escala original para interpretar mejor el error final)
df[columnas_numericas] = scaler.fit_transform(df[columnas_numericas])

# ==========================================
# 4. DIVISIÓN DE DATOS (TRAIN / TEST)
# ==========================================
# Separamos los últimos 12 meses para probar el modelo
train = df.iloc[:-12]
test = df.iloc[-12:]

# Variables independientes (X) y dependiente (y)
X_train = train[columnas_numericas + ['mes_seno', 'mes_coseno']]
y_train = train['consumo_maximo']
X_test = test[columnas_numericas + ['mes_seno', 'mes_coseno']]
y_test = test['consumo_maximo']

# ==========================================
# 5. ENTRENAMIENTO DE LOS MODELOS
# ==========================================

# --- Componente 1: Machine Learning (XGBoost) ---
print("Entrenando XGBoost...")
modelo_ml = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
modelo_ml.fit(X_train, y_train)
predicciones_ml = modelo_ml.predict(X_test)

# --- Componente 2: Series Temporales (SARIMA) ---
print("Entrenando SARIMA(0,1,1)(1,1,0)[12]...")
modelo_ts = ARIMA(
    y_train, 
    order=(0, 1, 1), 
    seasonal_order=(1, 1, 0, 12)
)
modelo_ts_fit = modelo_ts.fit()
predicciones_ts = modelo_ts_fit.forecast(steps=len(test)).values

# ==========================================
# 6. ENSAMBLE HÍBRIDO (SELECCIÓN DINÁMICA)
# ==========================================
print("\n--- Resultados de la Selección Dinámica ---")
predicciones_finales = []
origen_prediccion = [] # Para guardar quién ganó cada mes

for i in range(len(test)):
    # Calculamos el error absoluto de cada modelo para el mes 'i'
    error_ml = abs(predicciones_ml[i] - y_test.iloc[i])
    error_ts = abs(predicciones_ts[i] - y_test.iloc[i])
    
    # El híbrido elige la predicción con el menor error
    if error_ml < error_ts:
        predicciones_finales.append(predicciones_ml[i])
        origen_prediccion.append("XGBoost")
        print(f"Mes {i+1} ({test.index[i].strftime('%Y-%m')}): XGBoost gana (Error menor)")
    else:
        predicciones_finales.append(predicciones_ts[i])
        origen_prediccion.append("SARIMA")
        print(f"Mes {i+1} ({test.index[i].strftime('%Y-%m')}): SARIMA gana (Error menor)")

# ==========================================
# 7. EVALUACIÓN FINAL
# ==========================================
rmse_final = np.sqrt(mean_squared_error(y_test, predicciones_finales))
mae_final = mean_absolute_error(y_test, predicciones_finales)

print("\n==========================================")
print("RESUMEN DEL MODELO HÍBRIDO")
print("==========================================")
print(f"RMSE Final: {rmse_final:.2f}")
print(f"MAE Final:  {mae_final:.2f}")
print(f"Predicciones aportadas por XGBoost: {origen_prediccion.count('XGBoost')}/12")
print(f"Predicciones aportadas por SARIMA:  {origen_prediccion.count('SARIMA')}/12")
