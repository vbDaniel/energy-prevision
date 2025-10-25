import os
import json
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Configurações do pipeline
LOOKBACK = 24 * 7  # janela de 1 semana
FEATURES = [
    "Global_active_power",
    "temp",
    "holiday",
    "hour",
    "dayofweek",
    "month",
]
TARGET = "Global_active_power"


def load_best_model():
    """Carrega o melhor modelo salvo (.keras preferencial, fallback para .h5).
    Tenta primeiro os caminhos especificados pelo usuário e depois alternativas.
    """
    candidates = [
        "gru_energy_model_best.keras",  # raiz
        "models/gru_energy_model_best.h5",  # pasta models (pedido do usuário)
        "models/gru_energy_model_best.keras",  # pasta models
        "gru_energy_model_best.h5",  # raiz
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                if path.endswith(".h5"):
                    model = load_model(path, compile=False)
                else:
                    model = load_model(path)
                print(f"Modelo carregado: {path}")
                return model, path
            except Exception as e:
                print(f"Falha ao carregar modelo '{path}': {e}")
    raise FileNotFoundError(
        "Nenhum modelo encontrado. Certifique-se de que 'gru_energy_model_best.keras' ou 'models/gru_energy_model_best.h5' exista."
    )


def prepare_data():
    """Carrega dados de consumo e temperatura, cria features e normaliza com o scaler salvo."""
    # Consumo
    df = pd.read_csv(
        "data_power_consumption_sceaux.txt",
        sep=";",
        na_values="?",
        low_memory=False,
    )
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    df = df.drop(["Date", "Time"], axis=1)
    df = df.sort_values("datetime")

    cols = df.columns.drop("datetime")
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna()

    # Temperatura (JSON real)
    with open("data_temperature_sceaux.json", "r") as f:
        data_weather = json.load(f)

    weather_df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(data_weather["hourly"]["time"]),
            "temp": data_weather["hourly"]["temperature_2m"],
        }
    )
    weather_df["datetime"] = weather_df["datetime"].dt.tz_localize(None)

    # Merge por datetime (asof - mais próximo)
    df = pd.merge_asof(
        df.sort_values("datetime"),
        weather_df.sort_values("datetime"),
        on="datetime",
        direction="nearest",
    )

    # Recorte de teste igual ao script de treino
    test_df = df[df["datetime"] >= "2007-12-31"].copy()

    # Features exógenas
    test_df["dayofweek"] = test_df["datetime"].dt.weekday
    test_df["hour"] = test_df["datetime"].dt.hour
    test_df["month"] = test_df["datetime"].dt.month

    try:
        import holidays

        fr_holidays = holidays.France(years=[2006, 2007, 2008, 2009, 2010])
        fr_holiday_dates = set(fr_holidays)
        test_df["holiday"] = test_df["datetime"].dt.date.isin(fr_holiday_dates).astype(int)
    except Exception:
        # Fallback se pacote holidays não estiver disponível
        test_df["holiday"] = 0

    # Normalização com scaler salvo
    scaler = joblib.load("scaler_energy.pkl")
    test_X = test_df[FEATURES]
    test_X_scaled = scaler.transform(test_X)
    test_X = pd.DataFrame(test_X_scaled, columns=FEATURES, index=test_df.index).astype(np.float32)

    return test_df, test_X, scaler


def create_sequences(X: pd.DataFrame, lookback: int) -> np.ndarray:
    """Cria sequências com janela deslizante de tamanho lookback.
    Retorna N-L+1 sequências para alinhar com os targets do Keras timeseries_dataset.
    """
    Xs = []
    for i in range(len(X) - lookback + 1):
        Xs.append(X.iloc[i : i + lookback].values.astype(np.float32))
    return np.array(Xs, dtype=np.float32)


def invert_scale(y_scaled: np.ndarray, scaler, ref_X: np.ndarray) -> np.ndarray:
    """Inverte a escala do alvo usando o mesmo scaler do treino.
    ref_X é usado apenas para reconstruir o shape esperado pelo scaler.
    """
    zeros = np.zeros((len(y_scaled), ref_X.shape[1]), dtype=np.float32)
    zeros[:, 0] = np.array(y_scaled).ravel()
    inv = scaler.inverse_transform(zeros)[:, 0]
    return inv


def main():
    model, _model_path = load_best_model()
    test_df, test_X, scaler = prepare_data()

    # Sequências para inferência
    X_seq = create_sequences(test_X, LOOKBACK)

    # Predição (em escala normalizada)
    y_pred_scaled = model.predict(X_seq, verbose=0)

    # Targets do teste alinhados ao número de sequências
    y_test_scaled = test_X[TARGET].values[LOOKBACK - 1 :]

    # Reverter escala
    y_test_inv = invert_scale(y_test_scaled, scaler, test_X.values)
    y_pred_inv = invert_scale(y_pred_scaled, scaler, test_X.values)

    # Índice temporal e resíduos
    time_index = test_df["datetime"].iloc[LOOKBACK - 1 : LOOKBACK - 1 + len(y_test_inv)]
    res = y_test_inv - y_pred_inv

    # Salvar CSV em relatorios/previsoes_energia.csv
    out_dir = "relatorios"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "previsoes_energia.csv")

    df_pred = pd.DataFrame(
        {
            "datetime": time_index.values,
            "real_kW": y_test_inv,
            "previsto_kW": y_pred_inv,
            "residuo_kW": res,
        }
    )
    df_pred.to_csv(csv_path, index=False)
    print(f"Previsões salvas em CSV: {csv_path}")


if __name__ == "__main__":
    main()
