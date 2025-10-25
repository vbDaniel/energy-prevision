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

# Imports adicionais para otimização e visualização
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.dates as mdates
import os
import math
import time
try:
    import seaborn as sns
    if hasattr(sns, "set_theme"):
        sns.set_theme(style="whitegrid")
    else:
        sns.set(style="whitegrid")
except Exception:
    sns = None

# Detectar GPU e habilitar mixed precision quando disponível
gpu_devices = tf.config.list_physical_devices('GPU')
print("GPUs disponíveis:", gpu_devices)
if gpu_devices:
    mixed_precision.set_global_policy('mixed_float16')

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
fr_holiday_dates = set(fr_holidays)
train_df["holiday"] = train_df["datetime"].dt.date.isin(fr_holiday_dates).astype(int)
test_df["holiday"] = test_df["datetime"].dt.date.isin(fr_holiday_dates).astype(int)



# --------------------------
# 4. Seleção de variáveis
# --------------------------
# Variável alvo: consumo global ativo
target = "Global_active_power"

features = [target, "temp", "holiday", "hour", "dayofweek", "month"]

train_X = train_df[features]
test_X = test_df[features]

# Normalização (carregar scaler se existir)
scaler_path = "scaler_energy.pkl"
if os.path.exists(scaler_path):
    print("Carregando scaler salvo:", scaler_path)
    scaler = joblib.load(scaler_path)
    train_X_scaled = scaler.transform(train_X)
    test_X_scaled = scaler.transform(test_X)
else:
    scaler = MinMaxScaler()
    train_X_scaled = scaler.fit_transform(train_X)
    test_X_scaled = scaler.transform(test_X)

train_X = pd.DataFrame(train_X_scaled, columns=features, index=train_df.index).astype(np.float32)
test_X = pd.DataFrame(test_X_scaled, columns=features, index=test_df.index).astype(np.float32)

# --------------------------
# 5. Pipeline de janelamento com tf.data
# --------------------------
lookback = 24*7  # 1 semana de histórico
batch_size = 64  # manter mesmo parâmetro original

# Split de validação preservando histórico
val_ratio = 0.2
val_split_index = int(len(train_X) * (1 - val_ratio))
train_arr = train_X.values
train_tgt = train_X[target].values

# Conjuntos de treino e validação usando janelas completas
tr_input = train_arr[:val_split_index]
tr_targets = train_tgt[lookback-1:val_split_index]

ds_train = tf.keras.utils.timeseries_dataset_from_array(
    data=tr_input,
    targets=tr_targets,
    sequence_length=lookback,
    sequence_stride=1,
    shuffle=True,
    batch_size=batch_size
)
ds_train = ds_train.cache().prefetch(tf.data.AUTOTUNE)

val_input = train_arr[val_split_index:]
val_targets = train_tgt[val_split_index + lookback - 1:]

ds_val = tf.keras.utils.timeseries_dataset_from_array(
    data=val_input,
    targets=val_targets,
    sequence_length=lookback,
    sequence_stride=1,
    shuffle=False,
    batch_size=batch_size
)
ds_val = ds_val.cache().prefetch(tf.data.AUTOTUNE)

# Teste
te_input = test_X.values
te_targets = test_X[target].values[lookback-1:]

ds_test = tf.keras.utils.timeseries_dataset_from_array(
    data=te_input,
    targets=te_targets,
    sequence_length=lookback,
    sequence_stride=1,
    shuffle=False,
    batch_size=batch_size
)
ds_test = ds_test.cache().prefetch(tf.data.AUTOTUNE)

# --------------------------
# 6. Modelo GRU (compatível com mixed precision)
# --------------------------
model = Sequential([
    GRU(64, return_sequences=True, input_shape=(lookback, len(features))),
    Dropout(0.2),
    GRU(32),
    Dropout(0.2),
    Dense(1, dtype='float32')  # saída em float32 para estabilidade
])

# Compilar com tentativa de JIT (XLA)
try:
    model.compile(optimizer="adam", loss="mse", metrics=["mae"], jit_compile=True)
except TypeError:
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])  # fallback

model.summary()

# --------------------------
# 7. Treinar (ou carregar modelo salvo)
# --------------------------
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
class TimeEstimatorCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs, steps_per_epoch, verbose=True):
        super().__init__()
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.verbose = verbose
        self.batch_start_time = None
        self.seen_batches = 0
        self.total_batch_time = 0.0
        self.current_epoch = 0

    def on_train_batch_begin(self, batch, logs=None):
        self.batch_start_time = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch

    def on_train_batch_end(self, batch, logs=None):
        if self.batch_start_time is not None:
            dt = time.time() - self.batch_start_time
            self.total_batch_time += dt
            self.seen_batches += 1
        if self.verbose and self.seen_batches > 0:
            avg_step_time = self.total_batch_time / self.seen_batches
            remaining_steps_current_epoch = max(0, self.steps_per_epoch - (batch + 1))
            remaining_epochs = max(0, self.total_epochs - (self.current_epoch + 1))
            total_remaining_steps = remaining_steps_current_epoch + remaining_epochs * self.steps_per_epoch
            eta_seconds = total_remaining_steps * avg_step_time
            hrs = int(eta_seconds // 3600)
            mins = int((eta_seconds % 3600) // 60)
            secs = int(eta_seconds % 60)
            print(f"  [Estimador] Passo {batch+1}/{self.steps_per_epoch} | Época {self.current_epoch+1}/{self.total_epochs} | Tempo médio por step: {avg_step_time*1000:.1f} ms | ETA restante treinamento: {hrs}h {mins}m {secs}s   ", end="\r", flush=True)

    def on_epoch_end(self, epoch, logs=None):
        if self.seen_batches > 0 and self.verbose:
            avg_step_time = self.total_batch_time / self.seen_batches
            remaining_epochs = self.total_epochs - (epoch + 1)
            remaining_steps = remaining_epochs * self.steps_per_epoch
            eta_seconds = remaining_steps * avg_step_time
            hrs = int(eta_seconds // 3600)
            mins = int((eta_seconds % 3600) // 60)
            secs = int(eta_seconds % 60)
            print("\n", end="")
            print(f"\n[Estimador] Tempo médio por step: {avg_step_time:.3f}s | ETA restante: {hrs}h {mins}m {secs}s\n")

# calcular steps/epoch com base no número de janelas
n_train_sequences = tr_input.shape[0] - lookback + 1
steps_per_epoch = math.ceil(n_train_sequences / batch_size)

es = EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss')
rlrop = ReduceLROnPlateau(factor=0.5, patience=3, monitor='val_loss')
mc = ModelCheckpoint("gru_energy_model_best.keras", monitor='val_loss', save_best_only=True)
te_cb = TimeEstimatorCallback(total_epochs=20, steps_per_epoch=steps_per_epoch, verbose=True)

if os.path.exists("gru_energy_model_best.keras"):
    print("Carregando modelo salvo: gru_energy_model_best.keras")
    model = load_model("gru_energy_model_best.keras")
    history = None
elif os.path.exists("gru_energy_model_best.h5"):
    print("Carregando modelo salvo: gru_energy_model_best.h5 (legacy HDF5, carregando sem compilar)")
    model = load_model("gru_energy_model_best.h5", compile=False)
    # recompila o modelo após carregar do HDF5 para evitar problemas de desserialização de métricas
    try:
        model.compile(optimizer="adam", loss="mse", metrics=["mae"], jit_compile=True)
    except TypeError:
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])  # fallback
    # converter imediatamente o checkpoint para o formato .keras para evitar futuros problemas
    try:
        model.save("gru_energy_model_best.keras")
        print("Checkpoint convertido para formato .keras: gru_energy_model_best.keras")
    except Exception as e:
        print(f"Falha ao converter checkpoint para .keras: {e}")
    history = None
else:
    history = model.fit(
        ds_train,
        epochs=20,
        validation_data=ds_val,
        callbacks=[te_cb, es, rlrop, mc],
        verbose=1
    )

# --------------------------
# 8. Avaliar
# --------------------------
# Predição direta no dataset de teste
y_pred_scaled = model.predict(ds_test)
# Targets do teste alinhados ao número de sequências
y_test_scaled = te_targets

# Reverter escala
def invert_scale(y_scaled, ref_X):
    zeros = np.zeros((len(y_scaled), ref_X.shape[1]), dtype=np.float32)
    zeros[:,0] = np.array(y_scaled).ravel()
    inv = scaler.inverse_transform(zeros)[:,0]
    return inv

y_test_inv = invert_scale(y_test_scaled, test_X)
y_pred_inv = invert_scale(y_pred_scaled, test_X)

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
# Sempre salvar o modelo atual e o scaler
model.save("gru_energy_model.h5")  # salva a arquitetura e pesos
joblib.dump(scaler, "scaler_energy.pkl")  # salva o scaler

# --------------------------
# 10. Visualização aprimorada
# --------------------------
# Série temporal com datas
time_index = test_df["datetime"].iloc[lookback-1:lookback-1+len(y_test_inv)]

plt.figure(figsize=(16,6))
plt.plot(time_index, y_test_inv, label=f"Real (RMSE={rmse:.2f})")
plt.plot(time_index, y_pred_inv, label="Previsto")
plt.title(f"Previsão de Consumo de Energia (GRU)\nLookback={lookback}, Batch={batch_size}, Epochs=20")
plt.xlabel("Tempo (data/hora)")
plt.ylabel("Global_active_power (kW)")
plt.legend()
plt.grid(alpha=0.3)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.gcf().autofmt_xdate()
plt.show()

# Resíduos ao longo do tempo
res = y_test_inv - y_pred_inv
plt.figure(figsize=(16,4))
plt.plot(time_index, res, color='tab:red')
plt.title("Resíduos (Real - Previsto)")
plt.xlabel("Tempo")
plt.ylabel("kW")
plt.grid(alpha=0.3)
plt.show()

# Salvar previsões em CSV
out_dir = "relatorios"
os.makedirs(out_dir, exist_ok=True)
df_pred = pd.DataFrame({
    "datetime": time_index.values,
    "real_kW": y_test_inv,
    "previsto_kW": y_pred_inv,
    "residuo_kW": res
})
csv_path = os.path.join(out_dir, "previsoes_energia.csv")
df_pred.to_csv(csv_path, index=False)
print(f"Previsões salvas em CSV: {csv_path}")

# RMSE rolante em janela de 24h
window = 24
roll_rmse = np.sqrt(pd.Series(res).rolling(window).apply(lambda x: np.mean(x**2), raw=True))
plt.figure(figsize=(16,4))
plt.plot(time_index, roll_rmse, color='tab:blue')
plt.title("RMSE Rolante (24h)")
plt.xlabel("Tempo")
plt.ylabel("kW")
plt.grid(alpha=0.3)
plt.show()

# Scatter previsto vs real
plt.figure(figsize=(6,6))
plt.scatter(y_test_inv, y_pred_inv, alpha=0.5)
max_val = max(np.max(y_test_inv), np.max(y_pred_inv))
plt.plot([0, max_val], [0, max_val], 'r--', label='y=x')
plt.title("Previsto vs Real")
plt.xlabel("Real (kW)")
plt.ylabel("Previsto (kW)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Sazonalidade (opcional, se seaborn disponível)
if sns is not None:
    plt.figure(figsize=(10,5))
    sns.boxplot(x=test_df["hour"], y=test_df["Global_active_power"])
    plt.title("Distribuição do Consumo por Hora do Dia")
    plt.xlabel("Hora")
    plt.ylabel("Global_active_power (kW)")
    plt.show()

    plt.figure(figsize=(12,6))
    pivot = test_df.pivot_table(index="dayofweek", columns="hour", values="Global_active_power", aggfunc='mean')
    sns.heatmap(pivot, cmap='viridis')
    plt.title("Heatmap: Consumo médio por Dia da Semana x Hora")
    plt.xlabel("Hora")
    plt.ylabel("Dia da semana (0=Seg, ... 6=Dom)")
    plt.show()

