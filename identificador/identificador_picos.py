#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Identificador de horários de pico, dias da semana e meses de maior consumo energético.

Este script lê a base de consumo, agrega por hora/dia/mês, identifica picos por limiar
de percentil e salva resultados em JSON/CSV e gráficos em SVG.
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    if hasattr(sns, "set_theme"):
        sns.set_theme(style="whitegrid")
    else:
        sns.set(style="whitegrid")
except Exception:
    sns = None

DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_power_consumption_sceaux.txt")
OUTPUT_DIR = os.path.dirname(__file__)


def load_energy_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", na_values="?", low_memory=False)
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    df = df.drop(["Date", "Time"], axis=1)
    df = df.sort_values("datetime")
    # converter numérico e remover NaNs
    cols = df.columns.drop("datetime")
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna()
    # features temporais
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.weekday  # 0=Segunda
    df["month"] = df["datetime"].dt.month
    return df


def compute_aggregates(df: pd.DataFrame, target: str = "Global_active_power"):
    # médias por hora, dia da semana e mês
    by_hour = df.groupby("hour")[target].mean().sort_values(ascending=False)
    by_dow = df.groupby("dayofweek")[target].mean().sort_values(ascending=False)
    by_month = df.groupby("month")[target].mean().sort_values(ascending=False)
    return by_hour, by_dow, by_month


def detect_peaks(df: pd.DataFrame, target: str = "Global_active_power", percentile: float = 95.0):
    # pico global por percentil
    global_thresh = np.percentile(df[target].values, percentile)
    df["peak_global"] = (df[target] >= global_thresh).astype(int)
    # pico dentro do mesmo horário do dia
    hourly_thresh = df.groupby("hour")[target].transform(lambda x: np.percentile(x, percentile))
    df["peak_by_hour"] = (df[target] >= hourly_thresh).astype(int)
    # resumo de picos
    summary = {
        "percentile": percentile,
        "global_threshold": float(global_thresh),
        "peak_global_ratio": float(df["peak_global"].mean()),
        "peak_by_hour_ratio": float(df["peak_by_hour"].mean()),
        "peak_ratio_by_hour": df.groupby("hour")["peak_by_hour"].mean().to_dict(),
        "peak_ratio_by_dayofweek": df.groupby("dayofweek")["peak_global"].mean().to_dict(),
        "peak_ratio_by_month": df.groupby("month")["peak_global"].mean().to_dict(),
    }
    return df, summary


def save_results(by_hour, by_dow, by_month, summary, df_labeled):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # JSON
    results_json = {
        "top_hours_mean": by_hour.head(10).to_dict(),
        "top_days_mean": by_dow.head(7).to_dict(),
        "top_months_mean": by_month.head(12).to_dict(),
        "summary": summary,
    }
    with open(os.path.join(OUTPUT_DIR, "resultados_picos.json"), "w", encoding="utf-8") as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)
    # CSV rotulado
    df_labeled.to_csv(os.path.join(OUTPUT_DIR, "consumo_rotulado_picos.csv"), index=False)

    # Gráficos em SVG
    # Consumo médio por hora
    plt.figure(figsize=(10, 5))
    by_hour.sort_index().plot(kind="bar", color="tab:blue")
    plt.title("Consumo médio por hora do dia")
    plt.xlabel("Hora")
    plt.ylabel("Global_active_power (kW)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "consumo_por_hora.svg"))
    plt.close()

    # Consumo médio por dia da semana
    plt.figure(figsize=(10, 5))
    by_dow.sort_index().plot(kind="bar", color="tab:green")
    plt.title("Consumo médio por dia da semana (0=Seg, 6=Dom)")
    plt.xlabel("Dia da semana")
    plt.ylabel("Global_active_power (kW)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "consumo_por_dia_semana.svg"))
    plt.close()

    # Consumo médio por mês
    plt.figure(figsize=(10, 5))
    by_month.sort_index().plot(kind="bar", color="tab:orange")
    plt.title("Consumo médio por mês")
    plt.xlabel("Mês")
    plt.ylabel("Global_active_power (kW)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "consumo_por_mes.svg"))
    plt.close()

    # Heatmap (hora x dia) com seaborn (opcional)
    if sns is not None:
        pivot = df_labeled.pivot_table(index="dayofweek", columns="hour", values="Global_active_power", aggfunc="mean")
        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot, cmap="viridis")
        plt.title("Heatmap: Consumo médio por Dia da Semana x Hora")
        plt.xlabel("Hora")
        plt.ylabel("Dia da semana (0=Seg, ... 6=Dom)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "heatmap_dia_hora.svg"))
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Identificador de picos de consumo por hora/dia/mês")
    parser.add_argument("--percentile", type=float, default=95.0, help="Percentil para definição de pico (default: 95)")
    parser.add_argument("--data", type=str, default=DATA_FILE, help="Caminho para o arquivo de dados de consumo")
    args = parser.parse_args()

    df = load_energy_data(args.data)
    target = "Global_active_power"

    by_hour, by_dow, by_month = compute_aggregates(df, target=target)
    df_labeled, summary = detect_peaks(df.copy(), target=target, percentile=args.percentile)

    # Mostrar top categorias no console
    print("\nTop horas com maior consumo médio:")
    print(by_hour.head(10))
    print("\nTop dias da semana com maior consumo médio (0=Seg):")
    print(by_dow.head(7))
    print("\nTop meses com maior consumo médio:")
    print(by_month.head(12))

    save_results(by_hour, by_dow, by_month, summary, df_labeled)
    print(f"\nResultados salvos em: {OUTPUT_DIR}")
    print("Arquivos gerados: resultados_picos.json, consumo_rotulado_picos.csv, consumo_por_hora.svg, consumo_por_dia_semana.svg, consumo_por_mes.svg, heatmap_dia_hora.svg")


if __name__ == "__main__":
    main()