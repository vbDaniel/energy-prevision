#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classificador GRU para identificar a causa dominante do consumo energético
(com base em submedições) em cada instante, utilizando janelas temporais.

Classes: cozinha, lavanderia, aquecedor_agua_ar_condicionado, outros.

Entradas por passo: Global_active_power, sub1_kW, sub2_kW, sub3_kW, others_kW,
(opcional) Global_reactive_power, Voltage, Global_intensity, além de
recursos de tempo (hour, dayofweek, month, holiday).

Saídas:
- Modelo salvo (.keras) em models/classificador_causas_gru_best.keras
- Relatório e matriz de confusão (SVG)
- CSV com previsões por instante (classe e probabilidades)
"""
import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import joblib
import holidays

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_CONS = os.path.join(BASE_DIR, "data_power_consumption_sceaux.txt")
TEST_SIZE_DEFAULT = 0.2
OUT_DIR = os.path.join(BASE_DIR, "classificador")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

TARGET = "Global_active_power"
CAUSES = [
    "cozinha",
    "lavanderia",
    "aquecedor_agua_ar_condicionado",
    "outros",
]
CAUSE_TO_ID = {c: i for i, c in enumerate(CAUSES)}
ID_TO_CAUSE = {i: c for i, c in enumerate(CAUSES)}


def load_consumption(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", na_values="?", low_memory=False)
    # Dataset original usa "Date" e "Time"
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    df = df.drop(["Date", "Time"], axis=1)
    df = df.sort_values("datetime")
    cols = df.columns.drop("datetime")
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna()
    return df


def add_time_and_holidays(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.weekday
    df["month"] = df["datetime"].dt.month
    years = list(range(df["datetime"].dt.year.min(), df["datetime"].dt.year.max() + 1))
    fr_holidays = holidays.France(years=years)
    df["holiday"] = df["datetime"].dt.date.isin(set(fr_holidays)).astype(int)
    return df


def add_submeter_kw(df: pd.DataFrame) -> pd.DataFrame:
    cols = set(df.columns)
    s1_col = "Sub_metering_1" if "Sub_metering_1" in cols else ("sub_metering_1" if "sub_metering_1" in cols else None)
    s2_col = "Sub_metering_2" if "Sub_metering_2" in cols else ("sub_metering_2" if "sub_metering_2" in cols else None)
    s3_col = "Sub_metering_3" if "Sub_metering_3" in cols else ("sub_metering_3" if "sub_metering_3" in cols else None)
    if all([s1_col, s2_col, s3_col]):
        df["sub1_kW"] = df[s1_col] * 0.06
        df["sub2_kW"] = df[s2_col] * 0.06
        df["sub3_kW"] = df[s3_col] * 0.06
        df["others_kW"] = (df[TARGET] - (df["sub1_kW"] + df["sub2_kW"] + df["sub3_kW"]))
        df["others_kW"] = df["others_kW"].clip(lower=0.0)
    else:
        # Se não houver submedições, ainda permitimos treinar com TARGET
        df["sub1_kW"] = 0.0
        df["sub2_kW"] = 0.0
        df["sub3_kW"] = 0.0
        df["others_kW"] = df[TARGET]
    return df


def label_cause_from_row(row: pd.Series, min_share: float = 0.25, min_kw: float = 0.3) -> str:
    target_kw = float(row.get(TARGET, 0.0))
    s1 = float(row.get("sub1_kW", 0.0))
    s2 = float(row.get("sub2_kW", 0.0))
    s3 = float(row.get("sub3_kW", 0.0))
    others = float(row.get("others_kW", 0.0))
    contributions = [s1, s2, s3, others]
    idx = int(np.argmax(contributions))
    top = contributions[idx]
    share = (top / max(target_kw, 1e-6)) if target_kw > 0 else 0.0
    names = CAUSES
    if top >= min_kw and share >= min_share:
        return names[idx]
    else:
        return "outros"


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    # rolling de consumo para contexto
    df["consumo_rolling_3h"] = df[TARGET].rolling(window=3, min_periods=1).mean()
    df["consumo_rolling_24h"] = df[TARGET].rolling(window=24, min_periods=1).mean()
    return df


def prepare_sequences(df: pd.DataFrame, lookback: int, restrict_percentile: float | None = None) -> tuple:
    """Retorna (X, y, feature_cols, dates) já alinhados e prontos para criar datasets de sequência.
    Se restrict_percentile for definido, só mantém amostras com TARGET acima do percentil.
    """
    # Rotular cada instante
    df["cause"] = df.apply(label_cause_from_row, axis=1)
    y_series = df["cause"].map(CAUSE_TO_ID).fillna(CAUSE_TO_ID["outros"]).astype(int)

    # Opcional: restringir a instantes de maior consumo
    if restrict_percentile is not None:
        thresh = np.percentile(df[TARGET].values, restrict_percentile)
        mask = (df[TARGET].values >= thresh)
        df = df.loc[mask]
        y_series = y_series.loc[df.index]
        print(f"Filtrando instantes acima do percentil {restrict_percentile:.1f}: limiar {thresh:.3f} kW, amostras {len(df)}")

    feature_cols = [
        TARGET,
        "sub1_kW", "sub2_kW", "sub3_kW", "others_kW",
        "consumo_rolling_3h", "consumo_rolling_24h",
        # opcionais se existirem
        *(col for col in ["Global_reactive_power", "Voltage", "Global_intensity"] if col in df.columns),
        # tempo
        "hour", "dayofweek", "month", "holiday",
    ]

    # Montar matriz de features e normalizar
    X = df[feature_cols].ffill().bfill().values
    y = y_series.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Guardar scaler
    joblib.dump(scaler, os.path.join(MODELS_DIR, "classificador_causas_gru_scaler.pkl"))

    dates = df["datetime"].values
    return X_scaled, y, feature_cols, dates


def make_timeseries_datasets(X: np.ndarray, y: np.ndarray, lookback: int, test_size: float = 0.2, batch_size: int = 64):
    # Split temporal sem embaralhar
    n = X.shape[0]
    split = int(n * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Construir datasets de sequência (alinhando y ao final da janela)
    ds_train = tf.keras.utils.timeseries_dataset_from_array(
        data=X_train,
        targets=y_train,
        sequence_length=lookback,
        sequence_stride=1,
        sampling_rate=1,
        batch_size=batch_size,
    )
    ds_test = tf.keras.utils.timeseries_dataset_from_array(
        data=X_test,
        targets=y_test,
        sequence_length=lookback,
        sequence_stride=1,
        sampling_rate=1,
        batch_size=batch_size,
    )
    # Separar validação do final do treino
    val_take = max(1, int(0.1 * len(list(ds_train))))
    ds_val = ds_train.take(val_take)
    ds_train = ds_train.skip(val_take)

    return ds_train, ds_val, ds_test


def build_gru_model(input_shape: tuple[int, int], num_classes: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.GRU(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=False),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
        # jit_compile=False para evitar logs do XLA
    )
    return model


def train_and_evaluate(df: pd.DataFrame, lookback: int, restrict_percentile: float | None, batch_size: int, epochs: int, train_days: int):
    X, y, feature_cols, dates = prepare_sequences(df, lookback=lookback, restrict_percentile=restrict_percentile)

    # Definir janela de treino baseada em tempo
    start_date = pd.to_datetime(dates[0])
    train_end = start_date + pd.Timedelta(days=train_days)
    train_mask = pd.to_datetime(dates) < train_end

    X_train = X[train_mask]
    y_train = y[train_mask]

    if X_train.shape[0] < lookback + 10:
        raise ValueError("Poucos dados para treino após aplicar janela temporal. Aumente train_days ou reduza lookback.")

    # Dataset de treino completo (será particionado em treino/validação)
    ds_train_full = tf.keras.utils.timeseries_dataset_from_array(
        data=X_train,
        targets=y_train,
        sequence_length=lookback,
        sequence_stride=1,
        sampling_rate=1,
        batch_size=batch_size,
    )

    # Separar validação do final do treino (10% dos batches)
    n_batches_train_full = max(1, len(list(ds_train_full)))
    val_take = max(1, int(0.1 * n_batches_train_full))
    ds_val = ds_train_full.take(val_take)
    ds_train = ds_train_full.skip(val_take)

    # Determinar input_shape
    sample = next(iter(ds_train))
    xb, yb = sample
    input_shape = xb.shape[1:]  # (lookback, n_features)

    # Pesos de classe para balanceamento usando os alvos alinhados às janelas
    y_train_windows = y_train[lookback - 1:]
    classes = np.arange(len(CAUSES))
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_windows)
    class_weight = {int(cls): float(w) for cls, w in zip(classes, cw)}

    model = build_gru_model(input_shape=input_shape, num_classes=len(CAUSES))

    ckpt_path = os.path.join(MODELS_DIR, "classificador_causas_gru_best.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_acc", mode="max", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_acc", patience=5, restore_best_weights=True),
    ]

    history = model.fit(ds_train, validation_data=ds_val, epochs=epochs, class_weight=class_weight, verbose=1, callbacks=callbacks)

    # Avaliar em janelas de teste de 1 mês, 6 meses e 1 ano após o fim do treino
    test_spans = [("1m", 30), ("6m", 182), ("1y", 365)]
    results_by_span = {}
    for label, days in test_spans:
        test_start = train_end
        test_end = train_end + pd.Timedelta(days=days)
        test_mask = (pd.to_datetime(dates) >= test_start) & (pd.to_datetime(dates) < test_end)
        X_test = X[test_mask]
        y_test = y[test_mask]
        if X_test.shape[0] < lookback + 5:
            print(f"[Aviso] Janela de teste {label} insuficiente ({X_test.shape[0]} amostras). Ignorando.")
            continue
        ds_test = tf.keras.utils.timeseries_dataset_from_array(
            data=X_test,
            targets=y_test,
            sequence_length=lookback,
            sequence_stride=1,
            sampling_rate=1,
            batch_size=batch_size,
        )

        # Avaliar
        test_metrics = model.evaluate(ds_test, verbose=0)
        print(f"Teste {label} - Loss: {test_metrics[0]:.4f} | Acc: {test_metrics[1]:.4f}")

        # Obter previsões
        y_true = []
        y_pred = []
        y_prob = []
        for xb, yb in ds_test:
            probs = model.predict(xb, verbose=0)
            preds = np.argmax(probs, axis=1)
            y_pred.extend(list(preds))
            y_true.extend(list(yb.numpy()))
            y_prob.extend(list(probs))

        labels = list(range(len(CAUSES)))
        report = classification_report(y_true, y_pred, labels=labels, target_names=CAUSES, digits=3)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        print("\nRelatório de classificação (teste " + label + "):\n")
        print(report)

        # Salvar artefatos por janela
        # Matriz de confusão
        plt.figure(figsize=(8, 6))
        import seaborn as sns
        if hasattr(sns, "set_theme"):
            sns.set_theme(style="whitegrid")
        else:
            sns.set(style="whitegrid")
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CAUSES, yticklabels=CAUSES)
        plt.title(f"Matriz de confusão - GRU ({label})")
        plt.xlabel("Predito")
        plt.ylabel("Real")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"confusion_matrix_classificador_gru_{label}.svg"))
        plt.close()

        # CSV com previsões
        probs_df = pd.DataFrame(y_prob, columns=[f"prob_{c}" for c in CAUSES])
        preds_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
        out_pred = pd.concat([preds_df, probs_df], axis=1)
        out_pred.to_csv(os.path.join(OUT_DIR, f"predicoes_teste_classificador_gru_{label}.csv"), index=False)

        results_by_span[label] = {
            "metrics": test_metrics,
            "report": report,
            "confusion_matrix": cm,
        }

    # Salvar artefatos gerais
    final_path = os.path.join(MODELS_DIR, "classificador_causas_gru_final.keras")
    model.save(final_path)
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(os.path.join(OUT_DIR, "history_classificador_causas_gru.csv"), index=False)

    return {
        "model": model,
        "feature_cols": feature_cols,
        "lookback": lookback,
        "results_by_span": results_by_span,
        "train_end": train_end,
    }


def main():
    parser = argparse.ArgumentParser(description="Classificador GRU de causas dominantes do consumo")
    parser.add_argument("--data", type=str, default=DATA_CONS, help="Caminho para o arquivo de dados")
    parser.add_argument("--lookback", type=int, default=48, help="Tamanho da janela (passos)")
    parser.add_argument("--restrict_percentile", type=float, default=None, help="Opcional: filtrar instantes acima do percentil informado")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--min_share", type=float, default=0.25, help="Percentual mínimo do total para considerar a submedição como causa")
    parser.add_argument("--min_kw", type=float, default=0.3, help="Potência mínima (kW) da submedição para considerá-la causa")
    parser.add_argument("--train_days", type=int, default=365, help="Dias para treino antes de avaliar nas janelas (default: 365 dias)")
    args = parser.parse_args()

    df = load_consumption(args.data)
    df = add_time_and_holidays(df)
    df = add_submeter_kw(df)

    # Propagar limiares para rotulação
    global label_cause_from_row
    def label_cause_from_row(row: pd.Series, min_share: float = args.min_share, min_kw: float = args.min_kw) -> str:
        target_kw = float(row.get(TARGET, 0.0))
        s1 = float(row.get("sub1_kW", 0.0))
        s2 = float(row.get("sub2_kW", 0.0))
        s3 = float(row.get("sub3_kW", 0.0))
        others = float(row.get("others_kW", 0.0))
        contributions = [s1, s2, s3, others]
        idx = int(np.argmax(contributions))
        top = contributions[idx]
        share = (top / max(target_kw, 1e-6)) if target_kw > 0 else 0.0
        names = CAUSES
        if top >= min_kw and share >= min_share:
            return names[idx]
        else:
            return "outros"

    # Construir features complementares
    df = build_feature_frame(df)

    results = train_and_evaluate(df, lookback=args.lookback, restrict_percentile=args.restrict_percentile, batch_size=args.batch_size, epochs=args.epochs, train_days=args.train_days)
    print("\nTreino concluído. Artefatos gerados:")
    print(f"- Modelo best (checkpoint): {os.path.join(MODELS_DIR, 'classificador_causas_gru_best.keras')}")
    print(f"- Modelo final: {os.path.join(MODELS_DIR, 'classificador_causas_gru_final.keras')}")
    print(f"- CSV histórico: {os.path.join(OUT_DIR, 'history_classificador_causas_gru.csv')}")
    print(f"- CSV previsões de teste (1m/6m/1y): {os.path.join(OUT_DIR, 'predicoes_teste_classificador_gru_*.csv')}")
    print(f"- SVG matrizes de confusão (1m/6m/1y): {os.path.join(OUT_DIR, 'confusion_matrix_classificador_gru_*.svg')}")


if __name__ == "__main__":
    main()