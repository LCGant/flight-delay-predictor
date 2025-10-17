# Preditor de Atraso de Voos para o Brasil (Offline)

## 1) Introdução

O transporte aéreo brasileiro é marcado por sazonalidades fortes (férias, feriados móveis), heterogeneidade entre aeroportos e eventos imprevisíveis (clima/ATC). Nesse cenário, oferecer ao passageiro e ao planejador de viagens uma **estimativa de risco de atraso (>15 min na partida)**, mesmo sem dados operacionais em tempo real, tem valor prático: permite comparar alternativas e calibrar expectativas.

Este projeto implementa um **pipeline offline** que aprende padrões **estruturais e sazonais** a partir de histórico público (ANAC/INMET), gera probabilidades de atraso por voo e as expõe via **API local (FastAPI)** e por uma **extensão Chrome** que injeta o risco diretamente no Google Voos. O foco é a **qualidade do alerta**: preferimos menos alertas e mais confiáveis a muitos alertas barulhentos.

**Resumo executivo (TEST):** ROC AUC ≈ **0,73**, PR‑AUC ≈ **0,48** (baseline≈0,238), Precisão ≈ **0,59–0,60**, Recall ≈ **0,24**, com threshold escolhido por **`prec_at(0.6)`** (manter precisão ≥ 0,60).

---

## Sumário

1. Introdução
2. Estrutura do Projeto
3. Fontes de Dados & Download
4. Metodologia (alto nível)
5. Pipeline ETL (CLI)
6. Treinamento do Modelo
7. **Por que estas métricas? (ROC AUC, PR‑AUC, F1)**
8. API de Inferência (FastAPI)
9. Extensão Chrome (POC)
10. Métricas (TEST) e “por 100 voos”
11. Comparação com a Literatura (e limites)

    * 11.1) Aviso sobre comparações
12. Reprodutibilidade e Boas Práticas
13. **Features utilizadas e decisões de escopo**
14. Limitações & Próximos Passos
15. Referências
16. Licença

---

## 2) Estrutura do Projeto

```
.
├── data/
│   ├── processed/splits/       # train.parquet, val.parquet, test.parquet
│   └── models/                 # artefatos treinados (.joblib, .cbm)
├── reports/                    # metrics.json, summary_table.csv, quick_summary.txt, importances_*.csv
├── src/
│   ├── etl.py                  # utilidades/CLI de ETL
│   ├── features.py             # montagem/seleção de features
│   ├── model_train.py          # treinamento (RF / CatBoost)
│   └── model_utils.py          # auxiliares de IO/metrics/encoding
├── api/
│   └── serve_api.py            # FastAPI para predição local
└── chrome_extension/
    ├── manifest.json
    ├── background.js
    ├── content.js              # injeta “Atraso: XX%” no Google Voos
    └── overlay.css
```

---

## 3) Fontes de Dados & Download

### Voos — ANAC (Voo Regular Ativo — **VRA**)

* Página de histórico: [https://www.gov.br/anac/pt-br/assuntos/dados-e-estatisticas/historico-de-voos](https://www.gov.br/anac/pt-br/assuntos/dados-e-estatisticas/historico-de-voos)
* Metadados/portal dados abertos: [https://www.gov.br/anac/pt-br/acesso-a-informacao/dados-abertos/areas-de-atuacao/voos-e-operacoes-aereas/voo-regular-ativo-vra/62-voo-regular-ativo-vra](https://www.gov.br/anac/pt-br/acesso-a-informacao/dados-abertos/areas-de-atuacao/voos-e-operacoes-aereas/voo-regular-ativo-vra/62-voo-regular-ativo-vra)
* Catálogo/Dataset: [https://dados.gov.br/dados/conjuntos-dados/dadosabertos-areas-de-atuacao-voos-e-operacoes-aereas-voo-regular-ativo-vra](https://dados.gov.br/dados/conjuntos-dados/dadosabertos-areas-de-atuacao-voos-e-operacoes-aereas-voo-regular-ativo-vra)

### Clima — INMET

* Portal: [https://portal.inmet.gov.br/](https://portal.inmet.gov.br/)
* Catálogo de Estações Automáticas: [https://portal.inmet.gov.br/paginas/catalogoaut](https://portal.inmet.gov.br/paginas/catalogoaut)
* BDMEP (histórico; pode exigir cadastro): [https://bdmep.inmet.gov.br/](https://bdmep.inmet.gov.br/)

### Cadastro de Empresas Aéreas Nacionais — **PDA / Dados Abertos ANAC**

* Catálogo (dados.gov.br): [https://dados.gov.br/dados/conjuntos-dados/operador-aereo---empresas-aereas-nacionais](https://dados.gov.br/dados/conjuntos-dados/operador-aereo---empresas-aereas-nacionais)
* Metadados oficiais (Gov.br ANAC): [https://www.gov.br/anac/pt-br/acesso-a-informacao/dados-abertos/areas-de-atuacao/operador-aereo/empresas-aereas-nacionais/metadados-operador-aereo-empresas-aereas-nacionais](https://www.gov.br/anac/pt-br/acesso-a-informacao/dados-abertos/areas-de-atuacao/operador-aereo/empresas-aereas-nacionais/metadados-operador-aereo-empresas-aereas-nacionais)

### Aeródromos Públicos — **Lista oficial (versão atual)**

* Dataset (dados.gov.br): [https://dados.gov.br/dados/conjuntos-dados/aerodromos---lista-de-aerodromos-publicos-v2](https://dados.gov.br/dados/conjuntos-dados/aerodromos---lista-de-aerodromos-publicos-v2)

> **Guia prático (Colab) do tratamento de dados**
> Notebook com passo a passo da engenharia de features (ilustrativo, sem treino):
> [https://colab.research.google.com/drive/1JQiTQosvORPdl-O8zVLsmo0uOYnRWG3r?authuser=1#scrollTo=FzePxffmPIHw](https://colab.research.google.com/drive/1JQiTQosvORPdl-O8zVLsmo0uOYnRWG3r?authuser=1#scrollTo=FzePxffmPIHw)

---

## 🔗 3.1) Acesso à Pasta `data/` (Modelos e Artefatos)

Para reproduzir completamente o projeto — incluindo o **treinamento**, a **inferência offline** e a execução da **API local (FastAPI)** — é necessário ter acesso à pasta `data/`, que contém:

* **`data/processed/`** — conjuntos de treino, validação e teste (`.parquet`);
* **`data/models/`** — artefatos treinados (`.joblib`, `.cbm`);
* **`data/feature_store/`** *(opcional)* — cache intermediário de features.

📦 **Baixe a pasta completa no Google Drive:**

👉 [**Acessar pasta `data` no Google Drive**](https://drive.google.com/file/d/1uAVZTdl7ww-_uRZIcSWiL-t7UN7XxLND/view?usp=drive_link)

---

## 4) Metodologia (alto nível)

* **ETL & Features:** normalização de tipos; distância por Haversine; janelas históricas **anti‑vazamento** com `shift()`; proxies operacionais (congestionamento/buckets); calendário (mês, dia, feriados, período); codificação cíclica de hora (`sin/cos`).
* **Modelagem:** **Random Forest (sklearn/CPU)** e **CatBoost** (CPU/GPU); **balanceamento por rota/mês** (`route_quota`, `--target-pos 0.35`); threshold **`prec_at(0.6)`** escolhido em **VAL** e aplicado em **TEST**.

---

## 5) Pipeline ETL (CLI)

O `src/etl.py` expõe uma CLI com subcomandos; exemplos:

```bash
# 1) Unificar bases VRA (ANAC) e normalizar campos
python src/etl.py unify_all

# 2) Baixar/tratar clima (INMET) e integrar chaves
python src/etl.py weather

# 3) Montar o trainset tabular (join voos + clima + aeroportos + PDA)
python src/etl.py trainset

# 4) Gerar features históricas anti‑vazamento (shifts/rolagens)
python src/etl.py features

# 5) Criar splits temporais (train/val/test) em Parquet
python src/etl.py splits

# [atalho] Executar tudo em sequência
python src/etl.py all
```

**Saídas esperadas:** `data/processed/splits/{train,val,test}.parquet` com coluna‑alvo **`atraso15`**.

---

## 6) Treinamento do Modelo

### 6.1) Instalação

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 6.2) Random Forest (CPU — recomendado para servir via API)

```bash
python src/model_train.py \
  --model rf \
  --balance route_quota \
  --target-pos 0.35 \
  --rf-trees 1000 \
  --rf-max-depth 12 \
  --rf-min-leaf 20 \
  --rf-max-features sqrt \
  --th-policy prec_at --min-precision 0.6 \
  --seed 42
```

**Artefatos:** `data/models/rf_cpu_delay.joblib` • relatórios em `reports/`.

### 6.3) CatBoost (GPU opcional; CPU também funciona)

```bash
python src/model_train.py \
  --model catboost \
  --device gpu \
  --iters 5000 --lr 0.03 --depth 8 --l2 3.0 --od-wait 500 \
  --balance route_quota --target-pos 0.35 \
  --th-policy prec_at --min-precision 0.6 \
  --seed 42
```

---

## 7) Por que estas métricas? (ROC AUC, PR‑AUC, F1)

* **ROC AUC**: mede a **capacidade de ranquear** atrasos corretamente **sem escolher um limiar**. Útil para comparar modelos de forma geral.
* **PR‑AUC**: adequada para **classe desbalanceada** (atrasos ≈ 23,8%). A baseline de PR‑AUC é a **prevalência** (≈0,238); obter ≈**0,48** indica ~**2× melhor** que aleatório no regime precisão×recall.
* **F1 no limiar operacional**: resume **Precisão × Recall** **no ponto de decisão** usado em produção. Aqui **não** maximizamos F1; adotamos **`prec_at(0.6)`** para reduzir falsos alertas.

**Por que o F1≈0,34 e tudo bem?**
Com Precisão≈0,59 e Recall≈0,24, o **F1≈0,34** (F1=2PR/(P+R)). Esse valor é **baixo por decisão**: priorizamos **precisão** para reduzir ruído para o usuário. Em termos práticos (por 100 voos): emitimos ~9,8 alertas, **acertamos ~5,7** (TP), erramos ~4,0 (FP) e **deixamos ~18** atrasos sem alerta (FN).

**Quando o F1 subiria?**
(a) Relaxando a precisão (ex.: `prec_at(0.5)`), Recall ↑ e F1 tende a ↑ — com mais falsos alertas.
(b) Com **sinais operacionais em tempo real** (METAR/ATC/rotações), Recall ↑ sem sacrificar tanto a precisão.

> **Mensagem para apresentação (30–60s)**: “Usamos **ROC AUC** para comparação sem limiar, **PR‑AUC** porque a classe é **desbalanceada**, e **F1 no limiar real**. O F1 (~0,34) é **consequência planejada** de exigir **Precisão ≈ 0,60**: preferimos **poucos alertas e confiáveis**. Em TEST, AUC ≈ 0,73 e PR‑AUC ≈ 0,48 (baseline 0,238) — ~**2× melhor** que aleatório; por 100 voos, alertamos ~10 e acertamos ~6.”

---

## 8) API de Inferência (FastAPI)

```bash
uvicorn api.serve_api:app --host 0.0.0.0 --port 8000 --reload
curl http://127.0.0.1:8000/health
```

**Exemplo de requisição:**

```json
{
  "origin_iata": "GRU",
  "dest_iata": "GIG",
  "departure_iso": "2026-03-01T10:10",
  "airline_iata": "LA",
  "flight_number": null,
  "scenario": "clear"
}
```

---

## 9) Extensão Chrome (POC)

1. Garanta a API em `127.0.0.1:8000`
2. `chrome://extensions` → **Modo desenvolvedor** → **Carregar sem compactação** → `chrome_extension/`
3. Em `/search` e `/booking` do Google Voos, a extensão injeta **“Atraso: XX%”** nos cards/seleções
4. Logs com prefixo **[FDRB]**; cache por “chave de voo” (rota+data+hora+cia)

---

## 10) Métricas (TEST) e “por 100 voos”

**Classe positiva:** atraso > 15 min (prevalência ≈ **0.238** → baseline PR-AUC ≈ **0.238**).

| Modelo        | ROC AUC | PR-AUC |     F1 | Precisão | Recall |
| ------------- | ------: | -----: | -----: | -------: | -----: |
| Random Forest |  0.7267 | 0.4805 | 0.3422 |   0.5873 | 0.2415 |
| CatBoost      |  0.7272 | 0.4774 | 0.3413 |   0.5824 | 0.2414 |


**Por 100 voos (TEST):**

* **RF**: atrasos reais 23.8 | **alertas 9.8** → **TP 5.7**, **FP 4.0** | **FN 18.0**
* **CatBoost**: perfil praticamente idêntico (diferenças < 0.2 p.p.)

---

## 11) Comparação com a Literatura (e limites de comparabilidade)

| Estudo/Projeto           | Escopo & Dados                                             | Modelo             | Métricas Reportadas                              | Comentários                                                          |
| ------------------------ | ---------------------------------------------------------- | ------------------ | ------------------------------------------------ | -------------------------------------------------------------------- |
| **Este projeto (BR)**    | Brasil, vários aeroportos, **offline**                     | RF / CatBoost      | **ROC AUC ≈ 0.73**, **PR-AUC ≈ 0.48**, F1 ≈ 0.34 | Métricas adequadas ao desbalanceamento; limiar **prec_at(0.6)**      |
| arXiv: **2002.10254**    | EUA (heterogêneo), offline                                 | GBDT/LR (var.)     | **AUC ~ 0.70**                                   | Comparável via AUC; não enfatiza PR-AUC. Nosso AUC é **+2–3 p.p.**   |
| Stanford **CS229**       | EUA (1990–2008), subset (grandes cias/aeroportos), offline | NB / SVM / LR      | Sem ROC/PR padronizados; regressão: **MAE 8.22** | Em classificação tiveram dificuldades; comparação direta limitada    |
| Tang (2021)              | **Apenas JFK**, 1 ano                                      | Decision Tree      | **Accuracy 0.9778**                              | Métrica enganosa p/ desbalanceamento; provável overfitting/vazamento |
| Hatıpoğlu & Tosun (2024) | **1 aeroporto turco**, 3 anos                              | XGBoost            | **Accuracy 0.80**                                | Mais plausível; sem ROC/PR; escopo bem local                         |
| Dai (2024)               | Escopo não claro                                           | Híbrido complexo   | **Accuracy 0.972**                               | Muito elevado p/ contexto real; suspeita de overfitting/vazamento    |
| Sternberg et al. (2016)  | Brasil (ANAC), descritivo                                  | Padrões frequentes | —                                                | Análise exploratória; não comparável como classificador              |

### 11.1) Aviso sobre comparações (por que números “perfeitos” são suspeitos)

* **Métrica inadequada** (*accuracy* em base desbalanceada). Com ~23.8% de atrasos, prever sempre “sem atraso” já rende ~**76.2%**.
* **Escopo restrito** (um aeroporto/ano) vs **escopo Brasil** deste projeto.
* **Vazamento temporal** (janelas sem `shift()`, uso de variáveis pós‑evento).
* **Validação inadequada** (k‑fold aleatório em dados temporais).
* **Threshold “embelezado”** (max accuracy/F1) vs nossa política **`prec_at(0.6)`** focada em precisão operacional.

**Como ler nossos números:** **ROC AUC ≈ 0.73** e **PR‑AUC ≈ 0.48 (~2× baseline)** são **realistas** para um setup offline amplo; precisão ~0.60/recall ~0.24 refletem **postura conservadora**.

---

## 12) Reprodutibilidade e Boas Práticas

* **Seeds fixas** e divisão **temporal** dos splits.
* **Bundle RF** guarda encoders/medianas para inferência consistente.
* **Fallback CatBoost** GPU→CPU automático.
* Relatórios em `reports/` (tabelas + “por 100 voos”).

---

## 13) Features utilizadas e decisões de escopo

**Categorias de features (exemplos):**

* **Categóricas**: `icao_empresa`, `origem_icao`, `destino_icao`, `rota`.
* **Calendário/tempo**: `mes`, `dia_semana`, `hora_sin/cos`, `periodo_dia_id`, `is_feriado`, `is_weekend`.
* **Operacionais/proxies**: `dist_km`, `congestion_ratio`, `sched_block_min`, `airport_size_id`.
* **Históricas (coração do modelo)** — **anti‑vazamento** com `shift()`/janelas temporais:
  `hist_atraso_empresa_50`, `hist_atraso_num_voo_10`, `hist_atraso_rota_30`, `hist_std_rota_30`, `hist_atraso_origem_hora_30`, `atrasos_mesmo_aeroporto_1h_hist`.

**Por que **não** usamos “tudo” o que existia?**

* **Risco de vazamento temporal**: variáveis que olham o futuro ou janelas mal alinhadas distorcem o resultado (ex.: médias que incluem o próprio voo).
* **Cobertura baixa/ruído**: colunas com muitos `NaN` ou medidas instáveis (principalmente clima pontual) degradam generalização.
* **Colinearidade/redundância**: sinal duplicado (ex.: `hist_atraso_rota_30` já captura o efeito que outra coluna “similar” traria).
* **Indisponível no momento da decisão**: sinais **operacionais em tempo real** (METAR minuto‑a‑minuto, ATC, rotações) não entram no **setup offline** por **escopo**.

**Top features observadas (importância):**
`hist_atraso_empresa_50`, `hist_atraso_num_voo_10`, `atrasos_mesmo_aeroporto_1h_hist`, `hist_atraso_origem_hora_30`, `hist_atraso_rota_30`, `hist_std_rota_30`, além de `rota` e `icao_empresa`.

---

## 14) Limitações & Próximos Passos

**Limitações (offline):** sem METAR/TAF minuto‑a‑minuto, slots ATC, rotações de aeronave/tripulação, capacidade em tempo real. O modelo captura **padrões estruturais e sazonais**, mas não responde a **choques do dia**.

**Para subir o teto:**

* Clima operacional granular (teto/visibilidade/vento cruzado/rajada)
* Capacidade/fluxo ATC (NOTAM/slots)
* Estado de aeronave/tripulação
* Carga do aeroporto na janela da partida (arrivals bank, conexões)

---

## 15) Referências

[1] Tang, Y. (2021). *Airline Flight Delay Prediction Using Machine Learning Models*. 5th Int. Conf. on E‑Business and Internet. [https://doi.org/10.1145/3497701.3497725](https://doi.org/10.1145/3497701.3497725)
[2] Hatıpoğlu, I., & Tosun, Ö. (2024). *Predictive Modeling of Flight Delays at an Airport Using Machine Learning Methods*. Applied Sciences, 14(13), 5472. [https://doi.org/10.3390/app14135472](https://doi.org/10.3390/app14135472)
[3] Dai, M. (2024). *A hybrid machine learning-based model for predicting flight delay through aviation big data*. Scientific Reports, 14, 4603. [https://doi.org/10.1038/s41598-024-55217-z](https://doi.org/10.1038/s41598-024-55217-z)
[4] Sternberg, A., et al. (2016). *An analysis of Brazilian flight delays based on frequent patterns*. Transp. Res. Part E, 95, 282–298. [https://doi.org/10.1016/j.tre.2016.09.013](https://doi.org/10.1016/j.tre.2016.09.013)
[5] (Comparativo AUC) *arXiv:2002.10254* — referência de AUC ≈ 0.70.

---

## 16) Licença

Este projeto é distribuído sob a licença **MIT**. Veja `LICENSE`.
