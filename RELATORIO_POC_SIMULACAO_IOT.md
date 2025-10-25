# Relatório de Mudança — POC de Simulação de IoT com Prisma

Este documento descreve, de forma objetiva, o que precisa ser feito e exatamente o que será implementado para: (1) importar todo o arquivo de consumo de energia para o banco (Prisma/Postgres) e (2) simular dinamicamente a chegada de “dados novos” como se viessem de um dispositivo IoT, mantendo 3 anos “ocultos” inicialmente e liberando-os gradualmente para a aplicação.

## Objetivo

- Ingerir 100% do dataset `data_power_consumption_sceaux.txt` no banco.
- Garantir que a aplicação (Next + Prisma) passe a “ver” apenas 1 ano de dados inicialmente.
- Simular, de forma dinâmica, a chegada de novas leituras a partir dos 3 anos restantes, como se fossem dados IoT.
- Manter compatibilidade com o modelo de previsão (Python API) que trabalha com janelas horárias (lookback padrão 24\*7).

## Estado atual

- Prisma já possui os modelos: `PowerReading`, `TemperatureReading`, `Prediction`, `PredictionPoint`.
- Existe um seed (`prisma/seed.ts`) que importa temperatura (JSON, horário) e consumo (TXT). O seed atual lê o TXT linha a linha, sem agregação por hora.
- A API Next (`src/app/api/predict/run/route.ts`) usa os registros mais recentes de `PowerReading` para montar o payload para a API Python e salvar previsões.
- O modelo Python utiliza granularidade horária e lookback 168.

## Decisão de Dados (granularidade)

O arquivo TXT aparenta ter registros por minuto. Para compatibilidade com o modelo e as rotinas (lookback em horas), iremos:

- Agregar as leituras por hora (ex.: média horária de `Global_active_power` e demais métricas).
- A chave temporal por hora será o início da hora (ex.: `YYYY-MM-DDTHH:00:00`), em timezone local do dataset.

## Desenho da Solução

1. Banco de Dados (Prisma):

   - Manter `PowerReading` como a tabela “visível” para a aplicação.
   - Criar nova tabela `PowerReadingBuffer` (mesma estrutura) para armazenar os 3 anos “futuros” que serão publicados gradualmente.
   - Índices: `@@index([timestamp])` e `@@unique([timestamp])` também no buffer para evitar duplicidade.

   Exemplo de novos modelos (Prisma):

   ```prisma
   model PowerReadingBuffer {
     id                   Int      @id @default(autoincrement())
     timestamp            DateTime @db.Timestamp(6)
     globalActivePower    Float
     globalReactivePower  Float?
     voltage              Float?
     globalIntensity      Float?
     subMetering1         Float?
     subMetering2         Float?
     subMetering3         Float?
     createdAt            DateTime @default(now())
     queuedAt             DateTime @default(now())

     @@index([timestamp])
     @@unique([timestamp])
   }
   ```

2. Ingestão total do dataset:

   - Novo script de seed que:
     - Lê o TXT em modo streaming (linha a linha) para evitar uso elevado de memória.
     - Agrega leituras por hora e monta registros horários.
     - Separa 1 ano (inicial) para inserir em `PowerReading` e os 3 anos para `PowerReadingBuffer`.
     - Usa `createMany` com `skipDuplicates: true` em lotes de 1000.
   - Parâmetros configuráveis por `.env` (datas de corte, limites, batch size).

3. Simulação dinâmica (Publicação do buffer):

   - Criar um script “simulador IoT” (Node + Prisma) que:
     - Em intervalos configuráveis (ex.: a cada 60s), move o próximo(s) registro(s) do `PowerReadingBuffer` para `PowerReading`, em ordem crescente de `timestamp`.
     - Remove os registros do buffer após publicar (ou marca como publicados), garantindo que `PowerReading` receba dados “novos” contínuos.
     - Evita duplicatas via `createMany({ skipDuplicates: true })` ou `upsert`.
   - Variáveis de controle:
     - `SIMULATOR_RATE_MS` (intervalo entre publicações, ex.: 60000 ms)
     - `SIMULATOR_BATCH_SIZE` (quantidade de registros por ciclo, ex.: 1 ou 24)
     - `SIMULATOR_START_TS` (timestamp inicial opcional)

4. Integração com previsão (opcional):
   - Após publicar novas leituras, podemos (opcional) acionar o endpoint `POST /api/predict/run` para gerar previsões automaticamente.
   - Isso pode ser feito a cada ciclo ou de forma programada (ex.: a cada 24 registros publicados).

## O que será implementado exatamente

- Alterações de schema (Prisma):

  - Adição de `PowerReadingBuffer` com mesma estrutura de `PowerReading`.
  - Migração gerada e aplicada no Postgres.

- Scripts (Node/TS) no `prevision-frontend`:

  - `prisma/seed_power_hourly.ts`: seed streaming, agregação por hora e split em `PowerReading` + `PowerReadingBuffer`.
  - `scripts/simulator_power_buffer.ts`: simulador que publica do buffer para o principal.

- Ajustes no `package.json` (scripts npm):

  - `db:migrate` e `db:generate` (já existem).
  - `db:seed:power:hourly` para executar o novo seed.
  - `simulate:power` para iniciar o simulador.

- Configuração `.env` (exemplos):
  - `DATABASE_URL=postgresql://postgres:postgres@localhost:5435/energy_prevision`
  - `SEED_POWER_VISIBLE_YEARS=1` (anos “visíveis”) e `SEED_POWER_BUFFER_YEARS=3` (anos “no buffer”).
  - `SEED_POWER_START=2007-01-01T00:00:00` e `SEED_POWER_END=2010-12-31T23:59:59` (caso precise forçar recorte).
  - `SEED_POWER_BATCH_SIZE=1000`
  - `SIMULATOR_RATE_MS=60000`
  - `SIMULATOR_BATCH_SIZE=1`

## Passo a passo de execução (quando aprovado)

1. Subir Postgres:
   - `docker-compose up -d`
2. Preparar Prisma:
   - `cd prevision-frontend`
   - `npm run db:migrate`
   - `npm run db:generate`
3. Seed de temperatura (já existe, opcional):
   - `npm run db:seed` (garante horários de temperatura)
4. Novo seed de consumo horário:
   - `npm run db:seed:power:hourly` (com agregação por hora e split em 1 ano visível + 3 anos em buffer)
5. Iniciar simulador:
   - `npm run simulate:power` (publica dinamicamente do buffer para o principal)
6. Verificação:
   - Prisma Studio (`npm run prisma:studio`) para observar contagens nas tabelas
   - API Python (`http://localhost:8000/health`) e endpoint de previsão (`POST /api/predict/run`) para testar previsões com dados novos.

## Critérios de Aceite

- Todo o dataset (após agregação horária) está no banco: 1 ano em `PowerReading` e 3 anos em `PowerReadingBuffer`.
- A aplicação consome apenas o que está em `PowerReading`.
- O simulador publica registros do buffer em ordem temporal e a aplicação passa a enxergar “dados novos” automaticamente.
- Previsões podem ser executadas sobre os dados mais recentes sem erros de granularidade.

## Riscos e Mitigações

- Arquivo muito grande: usar leitura em streaming e inserção por lotes.
- Duplicatas de timestamp: `@@unique([timestamp])` e `skipDuplicates` nos inserts.
- Diferença de timezone: padronizar timestamps e documentar.
- Granularidade incorreta: validar média por hora e amostragem, ajustar se necessário.

## Perguntas para confirmação

1. Confirma que devemos trabalhar em granularidade HORÁRIA (agregação por hora)?
2. Confirma que o 1º ano ficará “visível” inicialmente e os 3 anos seguintes irão para o buffer?
3. Preferência por taxa de publicação no simulador (ex.: 1 registro/hora real, 1 registro/minuto, ou em lotes)?
4. Deseja acionar previsões automaticamente a cada publicação ou manter manual?

## Próximos passos

- Após aprovação deste documento, implementarei as migrações, criarei os scripts de seed e simulador, configurarei os scripts npm e validarei o fluxo end-to-end.

## Requisitos adicionais de UI/Dashboard para a POC

Para que os cards e gráficos reflitam o “tempo simulado” (dados publicados do buffer, p.ex. 2007), a aplicação passará a considerar como “agora” o maior timestamp disponível em `PowerReading`.

- Relógio de tempo simulado:

  - Fonte: `simulatedNow = max(timestamp) from PowerReading`.
  - Usado por todos os cards e gráficos ao invés de `Date.now()`.

- Card Consumo do Dia (substitui Consumo Atual):

  - Exibir o consumo diário acumulado em kWh do “dia corrente simulado”.
  - Cálculo: com dados horários, energia do dia ≈ soma de `globalActivePower (kW) * 1h` para todas as horas do dia simulado.
  - Query: `WHERE timestamp BETWEEN startOfDay(simulatedNow) AND endOfDay(simulatedNow)`.

- Card Alertas e Notificação:

  - Alerta: "Consumo acima do previsto — Importante: Você está 8% acima da meta deste mês".
  - Meta do mês: por padrão, o total previsto para o mês corrente simulado (soma de `previstoKw` em `PredictionPoint`). Alternativa: baseline do mês (média do mesmo mês no ano anterior).
  - Condição: `real_mensal > previsto_mensal * 1.08`.
  - Exibir também o desvio percentual atual em relação à meta.

- "Roda" do dashboard (atualização):

  - A cada publicação do simulador, atualizar:
    - Consumo energético até a data simulada vs previsão (acumulados YTD do mês e do ano corrente simulado).
    - Séries para 3 meses, 6 meses e 1 ano:
      - Linhas mensais (ou semanais) com `real` (PowerReading) e `previsto` (PredictionPoint) para o período retroativo a partir de `simulatedNow`.

- Card de Eficiência (acurácia no último ano):

  - Acurácia (%) no último ano simulado: `100 * (1 - MAPE)`, onde `MAPE = mean(|real - previsto| / max(real, ε))` usando `PredictionPoint` com `realKw` disponível.
  - Alternativamente, exibir também RMSE/MAE.
  - Período: últimos 12 meses até `simulatedNow`.

- Comparação mensal previsão vs real:

  - Mostrar barras com total mensal `real` vs `previsto`.
  - Fonte: agregação de `PowerReading` (energia mensal ≈ soma de kWh horários) e `PredictionPoint` (soma de `previstoKw` por mês). Período: 12 meses anteriores ao `simulatedNow`.

- Pico de consumo:
  - Integrar resultados do identificador (pasta `prevision-python/identificador`).
  - Expor endpoint na API Python que retorne o `resultados_picos.json` e, opcionalmente, recalcule picos para o período até `simulatedNow`.
  - Card exibe: hora/dia/mes com maiores médias de consumo; e um destaque para o maior pico recente.

### Endpoints/Consultas previstos no frontend

- GET `/api/simulated/now` → retorna `max(timestamp)` de `PowerReading`.
- GET `/api/metrics/consumo-dia?date=<simulatedNow>` → retorna kWh do dia.
- GET `/api/metrics/alertas-mes?month=<simulatedMonth>` → retorna status da meta e desvio percentual.
- GET `/api/metrics/series?range=3m|6m|12m` → séries agregadas (real vs previsto).
- GET `/api/metrics/eficiencia?range=12m` → acurácia (MAPE/MAE/RMSE) do último ano.
- GET `/api/metrics/comparacao-mensal?range=12m` → totais mensais real vs previsto.
- GET `/api/picos` (proxy para API Python) → resultados de picos.

### Critérios de Aceite (UI adicionais)

- Card de Consumo do Dia reflete imediatamente os dados publicados do buffer (via tempo simulado).
- Alerta de 8% acima da meta calcula e exibe corretamente com base no mês simulado.
- Gráficos de 3m/6m/12m mostram real vs previsto coerentes com `simulatedNow`.
- Card de Eficiência exibe acurácia (%) do último ano simulado.
- Comparação mensal e pico de consumo (via identificador) disponíveis e consistentes.
