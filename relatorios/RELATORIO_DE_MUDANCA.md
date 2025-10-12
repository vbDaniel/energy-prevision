# Relatório de Mudanças

Este documento descreve as otimizações e melhorias aplicadas ao projeto de previsão de energia, conforme orientações do arquivo PERFORMANCE_TREINO_E_VISUALIZACAO.md.

## Objetivo
- Otimizar o treinamento (desempenho e estabilidade) mantendo os mesmos parâmetros originais.
- Melhorar a qualidade das visualizações para avaliação do modelo.
- Garantir o salvamento e reuso do modelo e do scaler.

## Arquivos impactados
- energy-prevision.py (principal)
- predict_energy-using-model.py (uso do modelo e scaler salvos; sem alterações nesta etapa)

## Resumo das mudanças principais
- Configuração de mixed precision quando GPU está disponível, mantendo a camada de saída em float32 para estabilidade numérica.
- Migração do janelamento para tf.keras.utils.timeseries_dataset_from_array com cache/prefetch para eficiência.
- Correção do alinhamento entre janelas e alvos usando lookback-1, evitando off-by-one e garantindo métricas e gráficos consistentes.
- Modelo GRU mantido com mesma arquitetura e parâmetros originais; compilação com tentativa de jit_compile (XLA) com fallback automático.
- Inclusão de callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint (salvando melhor peso em gru_energy_model_best.h5).
- Carregamento automático do melhor modelo salvo quando existente, evitando re-treino desnecessário.
- Avaliação ajustada para usar y_test corretamente alinhado e reversão de escala para métricas no domínio original.
- Visualizações aprimoradas: série temporal com datas formatadas, resíduos, RMSE rolante, scatter Previsto vs Real e sazonalidade opcional com Seaborn.
- Salvamento garantido do modelo atual e do scaler: gru_energy_model.h5 e scaler_energy.pkl.

## Parâmetros mantidos
- lookback: 24*7 (1 semana de histórico)
- batch_size: 64
- epochs: 20
- Arquitetura do modelo: GRU(64, return_sequences=True) + Dropout(0.2) + GRU(32) + Dropout(0.2) + Dense(1)
- Otimizador: Adam
- Função de perda: MSE

## Pipeline de dados
- Escalonamento com MinMaxScaler; carregamento automático do scaler se scaler_energy.pkl existir.
- Criação de datasets de treino/validação/teste com timeseries_dataset_from_array, alinhando targets ao final das janelas (lookback-1).
- Pré-processamento com cache() e prefetch(tf.data.AUTOTUNE) para melhor throughput.

## Treinamento e callbacks
- EarlyStopping (patience=5, monitor='val_loss', restore_best_weights=True).
- ReduceLROnPlateau (factor=0.5, patience=3, monitor='val_loss').
- ModelCheckpoint (save_best_only=True, monitor='val_loss', arquivo: gru_energy_model_best.h5).
- Carregamento do melhor checkpoint ao iniciar, caso exista.

## Avaliação
- Predição no dataset de teste.
- Alinhamento de y_test ao número de janelas usando lookback-1.
- Inversão de escala para cálculo de MAE e RMSE no domínio original.

## Visualizações
- Série temporal das previsões com eixo de datas formatado e destaque do RMSE.
- Resíduos (Real - Previsto) ao longo do tempo.
- RMSE rolante em janela de 24h.
- Gráfico de dispersão Previsto vs Real, com linha de referência y=x.
- Sazonalidade (opcional) com Seaborn: boxplot por hora e heatmap dia da semana x hora.

## Salvamento de artefatos
- Modelo atual: gru_energy_model.h5
- Melhor modelo (checkpoint): gru_energy_model_best.h5
- Scaler: scaler_energy.pkl

## Como executar
1) Treino/avaliação:
- python3 energy-prevision.py

2) Predição com dados novos:
- python3 predict_energy-using-model.py

## Observações
- Mixed precision é ativada automaticamente quando uma GPU é detectada; a camada de saída permanece float32 para evitar issues de precisão/overflow.
- A compilação com jit_compile=True tenta habilitar XLA; caso não seja suportado, há fallback automático.
- As melhorias de alinhamento garantem que os gráficos e métricas estejam sincronizados com a série temporal.