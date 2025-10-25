#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classificador de causas prováveis dos maiores picos de consumo energético
em uma residência na França, com base em horário, dia da semana, mês,
temperatura e feriados.

Abordagem:
- Geração de rótulos fracos (heurísticos) para cada registro de pico
  (ex.: aquecimento no inverno/à noite, cozinha em horários de refeição,
  resfriamento em dias quentes, lavanderia em fins de semana, etc.)
- Treinamento de um classificador (RandomForest) para aprender padrões
  e generalizar a classificação de causa dos picos.
- Saídas: relatório de desempenho, matriz de confusão (SVG), importância
  das features (SVG), modelo salvo e CSV com previsões.

Observação: sem dados por aparelho (submedição), aplicamos heurísticas
compatíveis com hábitos típicos na França; ajuste conforme realidade local.
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import holidays

try:
    import seaborn as sns
    if hasattr(sns, "set_theme"):
        sns.set_theme(style="whitegrid")
    else:
        sns.set(style="whitegrid")
except Exception:
    sns = None

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_CONS = os.path.join(BASE_DIR, "data_power_consumption_sceaux.txt")
DATA_TEMP = os.path.join(BASE_DIR, "data_temperature_sceaux.json")
OUT_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

TARGET = "Global_active_power"

CAUSES = [
    "aquecimento",
    "cozinha",
    "resfriamento",
    "lavanderia",
    "agua_quente",
    "iluminacao_entretenimento",
    "outros",
]
CAUSE_TO_ID = {c: i for i, c in enumerate(CAUSES)}
ID_TO_CAUSE = {i: c for i, c in enumerate(CAUSES)}


def load_consumption(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", na_values="?", low_memory=False)
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    df = df.drop(["Date", "Time"], axis=1)
    df = df.sort_values("datetime")
    cols = df.columns.drop("datetime")
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna()
    return df


def load_temperature(path: str) -> pd.DataFrame:
    with open(path, "r") as f:
        data_weather = json.load(f)
    weather_df = pd.DataFrame({
        "datetime": pd.to_datetime(data_weather["hourly"]["time"]),
        "temp": data_weather["hourly"]["temperature_2m"],
    })
    weather_df["datetime"] = weather_df["datetime"].dt.tz_localize(None)
    return weather_df.sort_values("datetime")


def build_features(df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    # merge por datetime (nearest)
    merged = pd.merge_asof(
        df.sort_values("datetime"),
        weather_df.sort_values("datetime"),
        on="datetime",
        direction="nearest",
    )
    merged["hour"] = merged["datetime"].dt.hour
    merged["dayofweek"] = merged["datetime"].dt.weekday
    merged["month"] = merged["datetime"].dt.month
    # feriados França
    years = list(range(merged["datetime"].dt.year.min(), merged["datetime"].dt.year.max() + 1))
    fr_holidays = holidays.France(years=years)
    holiday_dates = set(fr_holidays)
    merged["holiday"] = merged["datetime"].dt.date.isin(holiday_dates).astype(int)
    # rolling para consumo
    merged["consumo_rolling_3h"] = merged[TARGET].rolling(window=3, min_periods=1).mean()
    merged["consumo_rolling_24h"] = merged[TARGET].rolling(window=24, min_periods=1).mean()
    # converter sub_metering (Wh/min) para kW médios: kW = Wh * 60 / 1000 = Wh * 0.06
    cols = set(merged.columns)
    s1_col = "Sub_metering_1" if "Sub_metering_1" in cols else ("sub_metering_1" if "sub_metering_1" in cols else None)
    s2_col = "Sub_metering_2" if "Sub_metering_2" in cols else ("sub_metering_2" if "sub_metering_2" in cols else None)
    s3_col = "Sub_metering_3" if "Sub_metering_3" in cols else ("sub_metering_3" if "sub_metering_3" in cols else None)
    if all([s1_col, s2_col, s3_col]):
        merged["sub1_kW"] = merged[s1_col] * 0.06
        merged["sub2_kW"] = merged[s2_col] * 0.06
        merged["sub3_kW"] = merged[s3_col] * 0.06
        merged["others_kW"] = (merged[TARGET] - (merged["sub1_kW"] + merged["sub2_kW"] + merged["sub3_kW"]))
        merged["others_kW"] = merged["others_kW"].clip(lower=0.0)
    else:
        merged["sub1_kW"] = 0.0
        merged["sub2_kW"] = 0.0
        merged["sub3_kW"] = 0.0
        merged["others_kW"] = merged[TARGET]
    return merged


def label_peak_cause(row: pd.Series) -> str:
    hour = int(row["hour"])
    dow = int(row["dayofweek"])  # 0=Seg
    month = int(row["month"])
    temp = float(row.get("temp", np.nan))
    holiday = int(row.get("holiday", 0))

    winter_months = {11, 12, 1, 2}
    summer_months = {6, 7, 8}

    # Heurísticas principais
    if (month in winter_months or (not np.isnan(temp) and temp <= 10.0)) and (hour >= 18 and hour <= 23):
        return "aquecimento"

    if ((hour >= 12 and hour <= 14) or (hour >= 19 and hour <= 21)):
        return "cozinha"

    if (month in summer_months or (not np.isnan(temp) and temp >= 26.0)) and (hour >= 12 and hour <= 17):
        return "resfriamento"

    if (dow in {5, 6}) and (hour >= 8 and hour <= 12):  # sábado(5), domingo(6)
        return "lavanderia"

    if (hour >= 6 and hour <= 8):
        return "agua_quente"

    if (hour >= 20 and hour <= 23) and (holiday == 1 or dow in {5, 6}):
        return "iluminacao_entretenimento"

    return "outros"


def label_peak_cause_from_submeters(row: pd.Series, min_share: float = 0.25, min_kw: float = 0.3) -> str:
    target_kw = float(row.get(TARGET, 0.0))
    s1 = float(row.get("sub1_kW", 0.0))
    s2 = float(row.get("sub2_kW", 0.0))
    s3 = float(row.get("sub3_kW", 0.0))
    others = float(row.get("others_kW", 0.0))
    contributions = [s1, s2, s3, others]
    names = ["cozinha", "lavanderia", "aquecedor_agua_ar_condicionado", "outros"]
    idx = int(np.argmax(contributions))
    top = contributions[idx]
    share = (top / max(target_kw, 1e-6)) if target_kw > 0 else 0.0
    if top >= min_kw and share >= min_share:
        return names[idx]
    else:
        return "outros"


def detect_peaks(df: pd.DataFrame, percentile: float = 95.0) -> pd.DataFrame:
    thresh = np.percentile(df[TARGET].values, percentile)
    df["is_peak"] = (df[TARGET] >= thresh).astype(int)
    return df, float(thresh)


def train_classifier(df: pd.DataFrame) -> dict:
    # apenas registros de pico
    peaks = df[df["is_peak"] == 1].copy()
    if len(peaks) < 100:
        print("[Aviso] Poucos registros de pico. O classificador pode ficar fraco.")

    # rótulo com base em submedição
    peaks["cause"] = peaks.apply(label_peak_cause_from_submeters, axis=1)
    # Mapear rótulos para inteiros, preenchendo desconhecidos como "outros"
    y_series = peaks["cause"].map(CAUSE_TO_ID).fillna(CAUSE_TO_ID["outros"]).astype(int)
    y = y_series.values

    feature_cols = [
        "hour", "dayofweek", "month", "temp", "holiday",
        "consumo_rolling_3h", "consumo_rolling_24h",
        "sub1_kW", "sub2_kW", "sub3_kW", "others_kW",
        *(col for col in ["Global_reactive_power", "Voltage", "Global_intensity"] if col in peaks.columns)
    ]
    X = peaks[feature_cols].ffill().bfill().values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Escolher estratificação apenas se todas as classes tiverem pelo menos 2 amostras
    class_counts = np.bincount(y, minlength=len(CAUSES))
    stratify_labels = y if (len(np.unique(y)) > 1 and np.min(class_counts) >= 2) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=stratify_labels
    )

    clf = RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    labels = list(range(len(CAUSES)))
    report = classification_report(y_test, y_pred, labels=labels, target_names=CASES_SAFE_NAMES(), digits=3)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    joblib.dump(clf, os.path.join(MODELS_DIR, "classificador_picos.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "classificador_picos_scaler.joblib"))

    fi = clf.feature_importances_
    save_feature_importance_svg(feature_cols, fi)
    save_confusion_matrix_svg(cm)

    return {
        "clf": clf,
        "scaler": scaler,
        "report": report,
        "confusion_matrix": cm,
        "peaks_labeled": peaks,
    }


def CASES_SAFE_NAMES():
    # nomes legíveis para relatório, alinhados ao CAUSES
    return [
        "Cozinha",
        "Lavanderia",
        "Aquecedor água / Ar-condicionado",
        "Outros",
    ]


def save_feature_importance_svg(features, importances):
    plt.figure(figsize=(10, 5))
    idx = np.argsort(importances)[::-1]
    plt.bar([features[i] for i in idx], importances[idx], color="tab:purple")
    plt.title("Importância das features para classificar picos")
    plt.ylabel("Importância relativa")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "feature_importance_picos.svg"))
    plt.close()


def save_confusion_matrix_svg(cm):
    plt.figure(figsize=(8, 6))
    if sns is not None:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CASES_SAFE_NAMES(), yticklabels=CASES_SAFE_NAMES())
    else:
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xticks(ticks=range(len(CASES_SAFE_NAMES())), labels=CASES_SAFE_NAMES(), rotation=30)
        plt.yticks(ticks=range(len(CASES_SAFE_NAMES())), labels=CASES_SAFE_NAMES())
    plt.title("Matriz de confusão - Classificação de causas dos picos (submedição)")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix_picos.svg"))
    plt.close()


def save_predictions_csv(peaks_labeled, clf, scaler):
    feature_cols = [
        "hour", "dayofweek", "month", "temp", "holiday",
        "consumo_rolling_3h", "consumo_rolling_24h",
        "sub1_kW", "sub2_kW", "sub3_kW", "others_kW",
        *(col for col in ["Global_reactive_power", "Voltage", "Global_intensity"] if col in peaks_labeled.columns)
    ]
    X_full = peaks_labeled[feature_cols].ffill().bfill().values
    X_full_scaled = scaler.transform(X_full)
    y_pred = clf.predict(X_full_scaled)
    peaks_labeled["cause_pred"] = [ID_TO_CAUSE.get(int(i), "outros") for i in y_pred]
    out_path = os.path.join(OUT_DIR, "picos_causas_previstas.csv")
    peaks_labeled.to_csv(out_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Classificador de causas dos picos de energia em residência na França")
    parser.add_argument("--percentile", type=float, default=95.0, help="Percentil para definição de pico (default: 95)")
    parser.add_argument("--data", type=str, default=DATA_CONS, help="Caminho para o arquivo de dados de consumo")
    parser.add_argument("--temp", type=str, default=DATA_TEMP, help="Caminho para o arquivo JSON de temperatura")
    args = parser.parse_args()

    df = load_consumption(args.data)
    weather = load_temperature(args.temp)
    merged = build_features(df, weather)

    merged, thresh = detect_peaks(merged, percentile=args.percentile)
    print(f"Limiar de pico (percentil {args.percentile}): {thresh:.3f} kW")

    results = train_classifier(merged)
    print("\nRelatório de classificação:\n")
    print(results["report"])

    save_predictions_csv(results["peaks_labeled"], results["clf"], results["scaler"])
    print(f"\nArquivos gerados em {OUT_DIR}: feature_importance_picos.svg, confusion_matrix_picos.svg, picos_causas_previstas.csv")
    print(f"Modelo salvo em {MODELS_DIR}: classificador_picos.joblib, classificador_picos_scaler.joblib")


if __name__ == "__main__":
    main()