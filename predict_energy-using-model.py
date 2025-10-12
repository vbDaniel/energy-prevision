import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# --------------------------
# 1. Carregar modelo e scaler
# --------------------------
try:
    model = load_model("gru_energy_model.keras")
except Exception:
    # legacy HDF5 fallback: load without compile to avoid Keras 3 deserialization issues
    model = load_model("gru_energy_model.h5", compile=False)
scaler = joblib.load("scaler_energy.pkl")

# --------------------------
# 2. Preparar novos dados
# --------------------------
# Exemplo: suponha que você tenha um novo dataframe "new_df"
# com as mesmas colunas ['Global_active_power', 'temp', 'holiday', 'hour', 'dayofweek', 'month']

lookback = 24*7
features = ['Global_active_power', 'temp', 'holiday', 'hour', 'dayofweek', 'month']

# Normalizar com o mesmo scaler treinado
new_X_scaled = scaler.transform(new_df[features])
new_X = pd.DataFrame(new_X_scaled, columns=features)

# Criar sequência com a mesma janela
def create_sequences(X, lookback):
    Xs = []
    for i in range(len(X) - lookback):
        Xs.append(X.iloc[i:i+lookback].values)
    return np.array(Xs)

X_new_seq = create_sequences(new_X, lookback)

# --------------------------
# 3. Fazer previsão
# --------------------------
y_pred_scaled = model.predict(X_new_seq)

# Reverter escala
def invert_scale(y_scaled, ref_X):
    zeros = np.zeros((len(y_scaled), ref_X.shape[1]))
    zeros[:, 0] = y_scaled.ravel()
    inv = scaler.inverse_transform(zeros)[:, 0]
    return inv

y_pred = invert_scale(y_pred_scaled, new_X)
print("Previsão do consumo de energia:", y_pred[-10:])  # mostra últimas previsões
