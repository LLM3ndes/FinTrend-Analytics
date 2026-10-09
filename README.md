# 📊 FinTrend-Analytics: Inteligência de Clientes & Pipeline de Churn Preditivo

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=for-the-badge&logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

Um framework completo de Análise de Dados & Machine Learning projetado para negócios de E-Commerce e SaaS analisarem o comportamento de compra de clientes, medirem a retenção por coortes e preverem o risco de churn utilizando Inteligência Artificial Explicável (XAI).

---

## 📌 Principais Funcionalidades & Módulos

A plataforma é dividida em três motores analíticos centrais:

1. **Segmentação de Clientes RFM (`fintrend/rfm.py`)**
   - Calcula os valores de **Recência**, **Frequência** e **Valor Monetário** (RFM) para cada cliente.
   - Computa pontuações normalizadas em quintis (1 a 5) para agrupar dinamicamente os usuários em níveis de negócio:
     - 🏆 **Campeões (VIP):** Maiores compradores com transações recentes e frequentes.
     - 💙 **Clientes Leais:** Compradores consistentes com alto engajamento.
     - ⚠️ **Em Risco (Pré-Churn):** Clientes historicamente leais cuja recência diminuiu.
     - 💤 **Inativos / Perdidos:** Baixo engajamento em todas as métricas.

2. **Matriz de Retenção de Coortes (`fintrend/cohort.py`)**
   - Rastreia coortes de aquisição mensal de clientes ao longo do tempo.
   - Calcula matrizes de decaimento da retenção percentual para avaliar o *product-market fit* de longo prazo e o LTV (*Lifetime Value*).

3. **Modelo Preditivo de Churn por Machine Learning (`fintrend/churn_model.py`)**
   - Utiliza um classificador **Random Forest** treinado em padrões de compras comportamentais.
   - Identifica clientes ativos em risco de churn (>90 dias de inatividade).
   - Fornece **Importância de Atributos (XAI)** para explicar quais métricas (ex.: ticket médio, queda de frequência) impulsionam a evasão de clientes.

---

## 📂 Arquitetura do Projeto

```text
FinTrend-Analytics/
├── main.py             # Orquestrador de pipeline & gerador de relatório executivo
├── requirements.txt    # Dependências do projeto
├── README.md           # Documentação
├── .gitignore          # Arquivos locais e de cache ignorados
├── data/
│   ├── generate_dataset.py # Gerador de dados sintéticos de e-commerce
│   └── sales_data.csv      # Dataset bruto de transações (3.000+ registros)
└── fintrend/
    ├── __init__.py         # Inicialização do pacote
    ├── rfm.py              # Lógica de métricas e segmentação RFM
    ├── cohort.py           # Motor de coortes mensais e matriz de retenção
    └── churn_model.py      # Treinamento de ML (Random Forest) e explicabilidade
