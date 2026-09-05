# Challenge ML — Score de risco de condução por telemetria

3º entregável — **Modelagem de Machine Learning**
Autor: Arthur Zeferino

Modelagem de um score de risco de condução a partir de dados de telemetria veicular
(`TB_MOTORISTA_VEICULO_EVENTO`), agregados por motorista / veículo / mês.

## O problema

O alvo é o `risco_proxy`: o percentil (0–100) de `alertas_por_km` winsorizado em P99, construído
na Fase 2. Por ser contínuo, o problema é tratado como **regressão**.

## Estrutura do notebook (`ML.ipynb`)

| Seção | Conteúdo |
|---|---|
| Fase 2 (células 1–15) | Carga, limpeza, engenharia de features, EDA e correlações |
| 1. Preparação dos dados | Seleção de features, amostragem estratificada, `StandardScaler` + One-Hot Encoding via `ColumnTransformer` |
| 2. Treinamento | Linear Regression, KNN e Random Forest (mínimo exigido: 2) |
| 3. Validação | Holdout 80/20 — MAE, RMSE e R² |
| 4. Tuning | `GridSearchCV` (5-fold) do `k` do KNN |
| 5. Interpretação | Importância de variáveis (Random Forest) e coeficientes (Linear Regression) |
| 6. Reavaliação sem vazamento | Re-treino dos 3 modelos removendo as features que compõem o alvo |

## Sobre a seção 6 (data leakage)

O R² = 1.000 da Random Forest na seção 3 **não** é sinal de um bom modelo. O `risco_proxy` é o
percentil de `alertas_por_km_w`, que por definição é a soma de `freada_por_km`, `aceleracao_por_km`,
`curva_por_km` e `vel_via_por_km` — exatamente quatro das features de entrada. O modelo não está
aprendendo risco, está re-derivando a fórmula do próprio alvo.

A seção 6 quantifica isso re-treinando os mesmos três modelos em três cenários:

* **A** — completo (baseline com vazamento)
* **B** — sem `vel_via_por_km_w` (a feature que concentra ~97,5% da importância)
* **C** — sem nenhuma das quatro taxas que compõem o alvo

O R² do cenário C é a medida do poder preditivo real, e é a importância de variáveis desse
cenário — não a da seção 5 — que responde de forma defensável "quais variáveis mais impactam o risco".

## Como executar

```bash
pip install -r requirements.txt
cp .env.example .env    # preencha as credenciais, ou deixe em branco para usar o CSV
jupyter lab ML.ipynb
```

### Fonte de dados

O notebook tenta primeiro o banco Postgres (RDS) e, em caso de falha, cai automaticamente para um
CSV local processado com a mesma SQL via DuckDB.

* **Via banco:** preencha `DB_HOST`, `DB_USER`, `DB_PASSWORD` etc. no `.env`.
* **Via CSV:** coloque `TB_MOTORISTA_VEICULO_EVENTO.csv` (separador `;`) na raiz do projeto, ou
  aponte `CSV_PATH` no `.env` para o caminho do arquivo.

O CSV **não** está versionado (ver `.gitignore`) por conter dados operacionais de clientes.

> As credenciais nunca ficam no notebook. O `.env` está no `.gitignore` e o `.env.example` traz
> apenas os nomes das variáveis.

## Ambiente

Python 3.10+ — bibliotecas em `requirements.txt`.
`SEED = 42` em toda a modelagem (amostragem, split e Random Forest) para reprodutibilidade.
