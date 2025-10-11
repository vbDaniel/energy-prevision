import json
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout

from tensorflow.keras.models import save_model, load_model
import joblib  # para salvar o scaler

import holidays

## python3 energy-prevision.py

# --------------------------
# 1. Carregar e preparar base
# --------------------------
# Leitura do CSV
df = pd.read_csv("data_power_consumption_sceaux.txt",
                 sep=";",
                 na_values="?",
                 low_memory=False)

df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
df = df.drop(['Date', 'Time'], axis=1)
df = df.sort_values("datetime")

cols = df.columns.drop("datetime")
df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
df = df.dropna()


#https://archive-api.open-meteo.com/v1/archive?latitude=48.7764&longitude=2.2903&start_date=2006-01-01&end_date=2010-12-31&hourly=temperature_2m,relative_humidity_2m,precipitation,rain&timezone=Europe%2FBerlin
# Simulação de temperatura 
# rng = np.random.default_rng(42)
# train_df["temp"] = 15 + 10*np.sin(2*np.pi*train_df["datetime"].dt.dayofyear/365) + rng.normal(0,2,len(train_df))
# test_df["temp"] = 15 + 10*np.sin(2*np.pi*test_df["datetime"].dt.dayofyear/365) + rng.normal(0,2,len(test_df))

# url = "https://archive-api.open-meteo.com/v1/archive?latitude=48.7764&longitude=2.2903&start_date=2006-01-01&end_date=2010-12-31&hourly=temperature_2m,relative_humidity_2m,precipitation,rain&timezone=Europe%2FBerlin"
# response = requests.get(url)
# data_weather = response.json()

# with open("weather_data.json", "w") as f:
#     json.dump(data_weather, f)


# --------------------------
# 3. Dados climáticos reais
# --------------------------
# Carrega o JSON
with open("data_temperature_sceaux.json", "r") as f:
    data_weather = json.load(f)

# Cria DataFrame com data/hora e temperatura
weather_df = pd.DataFrame({
    "datetime": pd.to_datetime(data_weather["hourly"]["time"]),
    "temp": data_weather["hourly"]["temperature_2m"]
})

# Ajustar timezone se necessário (a base original do consumo está em GMT+1 normalmente)
weather_df["datetime"] = weather_df["datetime"].dt.tz_localize(None)

# Junta temperatura real ao dataframe principal pelo datetime
df = pd.merge_asof(
    df.sort_values("datetime"),
    weather_df.sort_values("datetime"),
    on="datetime",
    direction="nearest"  # pega a leitura mais próxima, se faltar uma hora
)

# --------------------------
# 2. Recorte de dados
# --------------------------
# Treino: apenas primeiro ano
train_df = df[(df["datetime"] >= "2006-12-16") & (df["datetime"] < "2007-12-31")].copy()
# Teste: todo o resto
test_df = df[df["datetime"] >= "2007-12-31"].copy()



# --------------------------
# 3. Criar features exógenas
# --------------------------
# Dia da semana
train_df["dayofweek"] = train_df["datetime"].dt.weekday
test_df["dayofweek"] = test_df["datetime"].dt.weekday

# Hora do dia
train_df["hour"] = train_df["datetime"].dt.hour
test_df["hour"] = test_df["datetime"].dt.hour

# Mês
train_df["month"] = train_df["datetime"].dt.month
test_df["month"] = test_df["datetime"].dt.month

# Feriados (França)
fr_holidays = holidays.France(years=[2006,2007,2008,2009,2010])
train_df["holiday"] = train_df["datetime"].dt.date.astype("datetime64[ns]").isin(fr_holidays).astype(int)
test_df["holiday"] = test_df["datetime"].dt.date.astype("datetime64[ns]").isin(fr_holidays).astype(int)



# --------------------------
# 4. Seleção de variáveis
# --------------------------
# Variável alvo: consumo global ativo
target = "Global_active_power"

features = [target, "temp", "holiday", "hour", "dayofweek", "month"]

train_X = train_df[features]
test_X = test_df[features]

# Normalização
scaler = MinMaxScaler()
train_X_scaled = scaler.fit_transform(train_X)
test_X_scaled = scaler.transform(test_X)

train_X = pd.DataFrame(train_X_scaled, columns=features, index=train_df.index)
test_X = pd.DataFrame(test_X_scaled, columns=features, index=test_df.index)

# --------------------------
# 5. Função para janelar
# --------------------------
def create_sequences(X, lookback=24*7):  # 1 semana de histórico
    Xs, ys = [], []
    for i in range(len(X) - lookback):
        Xs.append(X.iloc[i:i+lookback].values)
        ys.append(X.iloc[i+lookback][target])
    return np.array(Xs), np.array(ys)

lookback = 24*7
X_train, y_train = create_sequences(train_X, lookback)
X_test, y_test = create_sequences(test_X, lookback)

# --------------------------
# 6. Modelo GRU
# --------------------------
model = Sequential([
    GRU(64, return_sequences=True, input_shape=(lookback, len(features))),
    Dropout(0.2),
    GRU(32),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

# --------------------------
# 7. Treinar
# --------------------------
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

# --------------------------
# 8. Avaliar
# --------------------------
y_pred = model.predict(X_test)

# Reverter escala
def invert_scale(y_scaled, ref_X):
    zeros = np.zeros((len(y_scaled), ref_X.shape[1]))
    zeros[:,0] = y_scaled.ravel()
    inv = scaler.inverse_transform(zeros)[:,0]
    return inv

y_test_inv = invert_scale(y_test, test_X)
y_pred_inv = invert_scale(y_pred, test_X)

# --------------------------
# 9. Métricas
# --------------------------
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))

print(f"MAE = {mae:.3f}, RMSE = {rmse:.3f}")

# --------------------------
# 11. Salvar modelo e scaler
# --------------------------
model.save("gru_energy_model.h5")  # salva a arquitetura e pesos
joblib.dump(scaler, "scaler_energy.pkl")  # salva o scaler


# --------------------------
# 10. Visualização
# --------------------------
plt.figure(figsize=(14,6))
plt.plot(y_test_inv[:500], label="Real")
plt.plot(y_pred_inv[:500], label="Previsto")
plt.legend()
plt.title("Previsão de Consumo de Energia com GRU + Exógenas")
plt.show()

