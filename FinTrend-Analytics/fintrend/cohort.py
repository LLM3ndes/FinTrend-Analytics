import numpy as np
import pandas as pd


def calculate_cohort_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
  """Gera a matriz de retenção percentual de clientes divididos por coortes mensais."""
  df_clean = df.copy()
  df_clean["transaction_date"] = pd.to_datetime(df_clean["transaction_date"])

  # 1. Extrai o primeiro dia do mês de cada transação
  def get_month(x):
    return pd.Timestamp(year=x.year, month=x.month, day=1)

  df_clean["TransactionMonth"] = df_clean["transaction_date"].apply(get_month)

  # 2. Identifica o mês da primeira compra de cada cliente (CohortMonth)
  grouping = df_clean.groupby("customer_id")["TransactionMonth"]
  df_clean["CohortMonth"] = grouping.transform("min")

  # 3. Calcula o tempo transcorrido em meses (CohortIndex)
  trans_year = df_clean["TransactionMonth"].dt.year
  trans_month = df_clean["TransactionMonth"].dt.month
  cohort_year = df_clean["CohortMonth"].dt.year
  cohort_month = df_clean["CohortMonth"].dt.month
  years_diff = trans_year - cohort_year
  months_diff = trans_month - cohort_month
  df_clean["CohortIndex"] = years_diff * 12 + months_diff

  # 4. Agrupa por CohortMonth e CohortIndex contando clientes únicos
  cohort_data = (
      df_clean.groupby(["CohortMonth", "CohortIndex"])["customer_id"]
      .nunique()
      .reset_index()
  )

  # 5. Tabela dinâmica (Pivot Table)
  cohort_counts = cohort_data.pivot(
      index="CohortMonth", columns="CohortIndex", values="customer_id"
  )

  # 6. Converte contagens absolutas para % de retenção sobre o tamanho original
  cohort_sizes = cohort_counts.iloc[:, 0]
  retention_matrix = cohort_counts.divide(cohort_sizes, axis=0) * 100

  # Formata os rótulos de linha para AAAA-MM
  retention_matrix.index = retention_matrix.index.strftime("%Y-%m")
  return retention_matrix, cohort_sizes