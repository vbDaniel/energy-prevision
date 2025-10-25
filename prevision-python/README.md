# Energy Prevision: Previsão e Classificação de Consumo Energético com GRU e RandomForest

Este repositório contém um conjunto de scripts para análise, identificação de picos, classificação de causas de consumo e previsão de consumo energético usando modelos de Machine Learning (GRU e RandomForest) aplicados à base de dados de uma residência na França.

O projeto está organizado em dois eixos principais:
- Previsão de consumo (regressão) com GRU: `energy-prevision.py`.
- Identificação e classificação de picos/causas (classificação) com RandomForest e GRU: `identificador/identificador_picos.py`, `identificador/classificador_picos.py` e `classificador/classificador_gru.py`.

Abaixo você encontra um relatório completo com os dados usados, objetivos, entradas/saídas, como executar e considerações importantes.

---

## Dados utilizados

1) `data_power_consumption_sceaux.txt`
- Formato: texto separado por `;`.
- Colunas principais usadas:
  - `Date`, `Time`: usadas para compor `datetime`.
  - `Global_active_power`: alvo principal (kW) para previsão/limiar de picos.
  - `Global_reactive_power`, `Voltage`, `Global_intensity`: usadas como features opcionais.
  - `Sub_metering_1`, `Sub_metering_2`, `Sub_metering_3` (ou variações em minúsculas): submedições em Wh/minuto (convertidas para kW médios via multiplicação por 0,06).

2) `data_temperature_sceaux.json`
- Estrutura (Open-Meteo):
  - `hourly.time`: timestamps horários.
  - `hourly.temperature_2m`: temperatura do ar (°C).
- Usado para enriquecer as features com temperatura real nas janelas de tempo (scripts em `identificador` e em `energy-prevision.py`).

---

## Estrutura do projeto

```
energy-prevision/
├── classificador/
│   └── classificador_gru.py
├── identificador/
│   ├── classificador_picos.py
│   └── identificador_picos.py
├── models/
│   ├── classificador_causas_gru_scaler.pkl
│   ├── classificador_picos.joblib
│   ├── classificador_picos_scaler.joblib
│   ├── gru_energy_model.h5
│   ├── gru_energy_model_best.h5
│   └── gru_energy_model_best.keras
├── relatorios/ (gráficos e relatórios diversos)
├── data_power_consumption_sceaux.txt
├── data_temperature_sceaux.json
├── energy-prevision.py
└── README.md
```

---

## Como preparar o ambiente

- Requisitos: Python 3.10+ recomendado.
- Dependências principais:
  - numpy, pandas, scikit-learn, matplotlib, seaborn (opcional), holidays, joblib, tensorflow.
- Instalação rápida:

```
pip install numpy pandas scikit-learn matplotlib seaborn holidays joblib tensorflow
```

Sugestão: executar os scripts a partir da raiz do projeto para simplificar caminhos.

---

## 1) Previsão de consumo: `energy-prevision.py`

Objetivo:
- Prever a série de `Global_active_power` (kW) com um modelo GRU usando histórico de consumo e features exógenas (temperatura, feriados, hora/dia/mês).

Entradas e processamento:
- Lê `data_power_consumption_sceaux.txt` (sep=";") e constrói `datetime` a partir de `Date`+`Time`.
- Integra `data_temperature_sceaux.json` (Open-Meteo) por `datetime` (nearest).
- Cria features temporais: `hour`, `dayofweek`, `month`, `holiday` (França, anos 2006–2010).
- Seleciona features: `[Global_active_power, temp, holiday, hour, dayofweek, month]`.
- Normaliza com `MinMaxScaler` e constrói janelas com `tf.keras.utils.timeseries_dataset_from_array`.

Modelo:
- Arquitetura GRU sequencial: duas camadas GRU (64 -> 32) com Dropout e saída densa para regressão.
- Treinamento com EarlyStopping, ReduceLROnPlateau e ModelCheckpoint (`gru_energy_model_best.keras`).

Avaliação e saídas:
- Métricas: MAE e RMSE no conjunto de teste.
- Artefatos gerados:
  - Modelos: `models/gru_energy_model.h5` (final) e `models/gru_energy_model_best.keras` (checkpoint).
  - Scaler: `scaler_energy.pkl`.
  - Gráficos: previsões vs. reais, resíduos, RMSE rolante e sazonalidade (se seaborn disponível), normalmente salvos/visíveis ao executar.

Como executar:

```
python3 energy-prevision.py
```

Observações:
- O script utiliza um recorte temporal fixo: treino ~ 1 ano e teste no restante, harmonizado com as datas presentes.
- Caso `gru_energy_model_best.h5` exista, é carregado e convertido para `.keras` automaticamente.

---

## 2) Identificador de picos: `identificador/identificador_picos.py`

Objetivo:
- Identificar horários de pico global de consumo (`Global_active_power`) por percentil, e gerar agregados por hora/dia/mês.

Entradas e processamento:
- Lê `data_power_consumption_sceaux.txt` e constrói `datetime`.
- Calcula médias por `hour`, `dayofweek` e `month`.
- Define picos com base em um percentil (ex.: 95º), tanto global quanto por hora do dia.

Saídas:
- JSON: `identificador/resultados_picos.json` com resumos de picos e agregados.
- CSV: `identificador/consumo_rotulado_picos.csv` (base rotulada com flags de pico).
- Gráficos SVG: `consumo_por_hora.svg`, `consumo_por_dia_semana.svg`, `consumo_por_mes.svg`, `heatmap_dia_hora.svg`.

Como executar:

```
python3 identificador/identificador_picos.py --percentile 95
```

---

## 3) Classificador de causas dos picos (RandomForest): `identificador/classificador_picos.py`

Objetivo:
- Classificar a causa provável dos maiores picos de consumo usando submedições e contexto temporal/climático.

Rotulagem e features:
- Converte `Sub_metering_1/2/3` de Wh/min para kW médios: kW = Wh × 0,06.
- Define `others_kW = Global_active_power - (sub1_kW + sub2_kW + sub3_kW)`, truncado em 0.
- Rotula causa do pico com base na submedição dominante que exceda limiares: `min_kw` (ex.: 0,3 kW) e `min_share` (ex.: 25%). Classes: `cozinha`, `lavanderia`, `aquecedor_agua_ar_condicionado`, `outros`.
- Cria features: tempo (`hour`, `dayofweek`, `month`, `holiday`), temperatura (`temp`), consumo rolante (`consumo_rolling_3h`, `consumo_rolling_24h`), submedições e (opcional) `Global_reactive_power`, `Voltage`, `Global_intensity`.

Treino e avaliação:
- Detecta picos pelo percentil (ex.: 95º).
- Treina `RandomForestClassifier` com `class_weight='balanced'` e normalização via `StandardScaler`.
- Relatório e matriz de confusão com rótulos explícitos.

Saídas:
- Modelos: `models/classificador_picos.joblib`, `models/classificador_picos_scaler.joblib`.
- Gráficos: `identificador/feature_importance_picos.svg`, `identificador/confusion_matrix_picos.svg`.
- CSV: `identificador/picos_causas_previstas.csv` (previsões e colunas de contexto).

Como executar:

```
python3 identificador/classificador_picos.py --percentile 95 \
  --data "data_power_consumption_sceaux.txt" \
  --temp "data_temperature_sceaux.json"
```

Notas:
- O script trata automaticamente colunas `Sub_metering_*` em maiúsculas/minúsculas.
- Usa `.ffill().bfill()` para lidar com faltantes e evita erros de estratificação quando há poucas amostras em alguma classe.

---

## 4) Classificador GRU de causas dominantes: `classificador/classificador_gru.py`

Objetivo:
- Classificar, a cada instante, qual é a causa dominante de consumo entre as classes: `cozinha`, `lavanderia`, `aquecedor_agua_ar_condicionado`, `outros`.

Rotulagem e features:
- Rotulagem automática por submedições (dominância):
  - Converte `Sub_metering_1/2/3` para kW (× 0,06) e calcula `others_kW`.
  - Atribui a classe cuja submedição contribui mais e supera `min_kw` e `min_share` do total.
- Features por passo da sequência:
  - `Global_active_power`, `sub1_kW`, `sub2_kW`, `sub3_kW`, `others_kW`.
  - Consumo rolante: `consumo_rolling_3h`, `consumo_rolling_24h`.
  - Opcionais: `Global_reactive_power`, `Voltage`, `Global_intensity` (se presentes).
  - Tempo: `hour`, `dayofweek`, `month`, `holiday`.
- Normalização com `StandardScaler` e criação de janelas de tamanho `lookback`.

Treino e avaliação baseados em tempo:
- Treino: janela de `--train_days` dias (padrão: 365).
- Teste: três janelas consecutivas após o treino: 1 mês (30 dias), 6 meses (182 dias) e 1 ano (365 dias).
- Balanceamento: `compute_class_weight` nos alvos alinhados às janelas.
- Callbacks: checkpoint do melhor (`val_acc`) e EarlyStopping.

Saídas:
- Modelos: `models/classificador_causas_gru_best.keras` (melhor) e `models/classificador_causas_gru_final.keras` (final).
- Scaler: `models/classificador_causas_gru_scaler.pkl`.
- CSV: `classificador/history_classificador_causas_gru.csv` (histórico de treino), `classificador/predicoes_teste_classificador_gru_{1m,6m,1y}.csv` (previsões e probabilidades).
- Gráficos: `classificador/confusion_matrix_classificador_gru_{1m,6m,1y}.svg`.

Como executar (exemplos):

```
# Treinar 1 ano e avaliar nas janelas 1m, 6m e 1y
python3 classificador/classificador_gru.py --lookback 48 --batch_size 64 --epochs 20 --train_days 365

# Ajustar sensibilidade da rotulagem por submedição
python3 classificador/classificador_gru.py --min_share 0.25 --min_kw 0.3

# Focar em instantes de maior consumo (ex.: top 95º percentil)
python3 classificador/classificador_gru.py --restrict_percentile 95
```

Notas:
- As colunas `Sub_metering_*` são reconhecidas em maiúsculas/minúsculas.
- Se submedições não existirem, o script ainda funciona, rotulando tudo como `outros`.
- `others_kW` é truncado em 0 para evitar valores negativos.

---

## Replicando o experimento de 1 ano de treino e múltiplos testes

1. Identificação de picos e estatísticas básicas:
```
python3 identificador/identificador_picos.py --percentile 95
```

2. Classificação de causas dos picos (RandomForest) com submedições:
```
python3 identificador/classificador_picos.py --percentile 95 \
  --data "data_power_consumption_sceaux.txt" \
  --temp "data_temperature_sceaux.json"
```

3. Classificação GRU por causa dominante com 1 ano de treino:
```
python3 classificador/classificador_gru.py --train_days 365 --lookback 48 --epochs 20 --batch_size 64
```

4. Previsão contínua da série de consumo com GRU (regressão):
```
python3 energy-prevision.py
```

---

## Considerações e limitações

- Rotulagem por submedições é heurística e depende de limiares (`min_share`, `min_kw`). Ajuste conforme realidade dos equipamentos.
- Qualidade dos modelos depende de cobertura temporal de dados, sazonalidade e presença de feriados/temperatura.
- Em períodos com poucas amostras por classe, a estratificação é desabilitada para evitar erros e o desempenho pode variar.
- Recomenda-se revisar e calibrar janelas (`lookback`) e métricas/artefatos gerados (CSV, SVG) para o seu caso de uso.

---

## Backend FastAPI: como rodar a API para integração com Next.js

Esta API expõe endpoints para consumir artefatos (CSV/JSON/SVG) e realizar previsões/classificações em tempo real a partir dos modelos treinados.

Pré-requisitos:
- Python 3.10 ou 3.11 recomendado (TensorFlow 2.15 tem suporte mais amplo nessas versões).
- macOS/Linux/Windows com suporte a virtualenv.

Passo a passo:
1) Crie e ative um ambiente virtual na raiz do projeto:
```
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
# No Windows (PowerShell): .venv\Scripts\Activate.ps1
```
2) Instale as dependências:
```
pip install --upgrade pip
pip install -r requirements.txt
```
Se ocorrer erro de instalação do TensorFlow na sua versão de Python (ex.: 3.13), use Python 3.10/3.11 em um venv ou instale apenas os pacotes leves para testar os endpoints que não dependem de TensorFlow:
```
pip install fastapi uvicorn[standard] pydantic numpy pandas scikit-learn joblib holidays
```
3) Inicie o servidor FastAPI:
```
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Se o comando uvicorn não for encontrado, instale-o:
```
pip install uvicorn[standard]
```

Rotas principais:
- GET /health: verificação de saúde do servidor.
- GET /artifacts: lista arquivos disponíveis nos diretórios relatorios/, identificador/ e models/.
- GET /reports/previsoes-energia: retorna o conteúdo de relatorios/previsoes_energia.csv em JSON.
- GET /identificador/picos-causas: retorna o conteúdo de identificador/picos_causas_previstas.csv em JSON.
- GET /identificador/resultados-picos: retorna o conteúdo de identificador/resultados_picos.json.
- POST /predict/energy: previsão de consumo com GRU a partir de registros (lookback configurável).
- POST /classify/causas-gru: classificação de causa dominante com o classificador GRU.
- POST /classify/picos-rf: classificação de picos usando RandomForest.

Arquivos estáticos (montados automaticamente):
- /static/relatorios → conteúdos de relatorios/ (ex.: previsoes_energia.csv, gráficos PNG/SVG).
- /static/identificador → conteúdos de identificador/ (ex.: confusion_matrix_picos.svg, picos_causas_previstas.csv).
- /static/models → conteúdos de models/ (ex.: checkpoints .keras/.h5, scalers .pkl/.joblib).

Modelos e scalers esperados:
- Previsão de energia (GRU): models/gru_energy_model_best.keras (ou gru_energy_model_best.h5) e scaler_energy.pkl.
- Classificador GRU de causas: models/classificador_causas_gru_best.keras e models/classificador_causas_gru_scaler.pkl.
- Classificador de picos (RandomForest): models/classificador_picos.joblib e models/classificador_picos_scaler.joblib.

Integração com Next.js (exemplos):
- Consumir artefatos estáticos:
  - IMG: <img src="http://localhost:8000/static/identificador/confusion_matrix_picos.svg" />
  - CSV: fetch("http://localhost:8000/static/relatorios/previsoes_energia.csv")
- Consumir endpoints JSON:
  - fetch("http://localhost:8000/reports/previsoes-energia")
  - fetch("http://localhost:8000/identificador/resultados-picos")
- Previsão em tempo real (POST /predict/energy): envie um corpo JSON com `records` contendo as features `Global_active_power` e, opcionalmente, `datetime` para preenchimento automático de `hour`, `dayofweek`, `month` e `holiday`.

Dicas:
- Inicie o servidor na raiz do projeto para que as rotas estáticas encontrem os diretórios.
- Evite carregar TensorFlow na inicialização: a API faz importação lazy; modelos pesados são carregados apenas quando você chama os endpoints correspondentes.
- Em caso de erro de versão do TensorFlow, rode apenas os endpoints leves (artefatos e RandomForest) enquanto ajusta o ambiente.

---

### Solução de problemas (instalação e execução)

- zsh e extras do uvicorn: em zsh, colchetes são interpretados como glob. Sempre cite ou escape `uvicorn[standard]`.
```
pip install "uvicorn[standard]"
pip install 'uvicorn[standard]'
pip install uvicorn\[standard]
```

- Use sempre o pip do venv (evita o erro "externally-managed-environment"):
```
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
```

- Instalação segura dos pacotes mínimos (API leve, sem TensorFlow):
```
python3 -m pip install fastapi "uvicorn[standard]" pydantic numpy pandas scikit-learn joblib holidays
```

- Compatibilidade Python/TensorFlow:
  - Recomendo Python 3.10 ou 3.11 para usar TensorFlow 2.15 com numpy 1.26.x.
  - Se estiver em Python 3.13, suba apenas os endpoints leves (artefatos e RandomForest) e depois ajuste o ambiente para GRU.

- Evitar pré-releases de numpy: se seu pip estiver pegando versões `rc` (pré-release), garanta que você não está usando flags de pré-release e prefira versões estáveis (ex.: `numpy==2.2.3` se não usar TensorFlow).

- Comandos completos (copiar e colar):
```
# venv
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip

# instalação mínima (zsh)
python3 -m pip install fastapi "uvicorn[standard]" pydantic numpy pandas scikit-learn joblib holidays

# subir servidor
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# testar
# http://localhost:8000/health
```

---

## Créditos e licença

Projeto acadêmico do CEFET-MG (11º Período) para TCC, com foco em análise e previsão de consumo energético. Utilize e adapte conforme necessidade, citando a fonte.
