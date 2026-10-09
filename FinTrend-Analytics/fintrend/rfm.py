import numpy as np
import pandas as pd


def calculate_rfm_metrics(
    df: pd.DataFrame, reference_date: str | None = None
) -> pd.DataFrame:
  """Calcula as métricas brutas de Recência, Frequência e Valor Monetário por cliente."""
  df_clean = df.copy()
  df_clean["transaction_date"] = pd.to_datetime(df_clean["transaction_date"])
  if reference_date is None:
    reference_date = df_clean["transaction_date"].max() + pd.Timedelta(days=1)
  else:
    reference_date = pd.to_datetime(reference_date)

  rfm = (
      df_clean.groupby("customer_id")
      .agg({
          "transaction_date": lambda x: (reference_date - x.max()).days,
          "transaction_id": "count",
          "amount": "sum",
      })
      .reset_index()
  )
  rfm.columns = ["customer_id", "recency", "frequency", "monetary"]
  return rfm


def calculate_rfm_scores(rfm_df: pd.DataFrame) -> pd.DataFrame:
  """Atribui notas de 1 a 5 para Recência, Frequência e Valor Monetário usando quantis."""
  rfm = rfm_df.copy()
  # Recência: Menos dias de inatividade = nota maior (5 é a melhor pontuação)
  rfm["R_score"] = pd.qcut(
      rfm["recency"].rank(method="first"), q=5, labels=[5, 4, 3, 2, 1]
  )
  # Frequência: Mais compras = nota maior
  rfm["F_score"] = pd.qcut(
      rfm["frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]
  )
  # Monetário: Maior valor total gasto = nota maior
  rfm["M_score"] = pd.qcut(
      rfm["monetary"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]
  )

  # Converte tipos de dados para inteiros
  rfm["R_score"] = rfm["R_score"].astype(int)
  rfm["F_score"] = rfm["F_score"].astype(int)
  rfm["M_score"] = rfm["M_score"].astype(int)

  # Score concatenado (Ex: '555' para o cliente perfeito)
  rfm["RFM_Cell"] = (
      rfm["R_score"].astype(str)
      + rfm["F_score"].astype(str)
      + rfm["M_score"].astype(str)
  )
  return rfm


def segment_customers(rfm_df: pd.DataFrame) -> pd.DataFrame:
  """Categoriza os clientes em segmentos estratégicos de negócio com base nas notas RFM."""
  rfm = rfm_df.copy()

  def assign_segment(row):
    r, f, m = row["R_score"], row["F_score"], row["M_score"]
    if r >= 4 and f >= 4 and m >= 4:
      return "Campeões (VIP)"
    elif r >= 3 and f >= 3:
      return "Clientes Leais"
    elif r >= 3 and f <= 2:
      return "Promissores / Novos"
    elif r <= 2 and f >= 3:
      return "Em Risco (Pré-Churn)"
    elif r <= 2 and f <= 2 and m >= 3:
      return "Não Podemos Perdê-los"
    else:
      return "Inativos / Perdidos"

  rfm["segment"] = rfm.apply(assign_segment, axis=1)
  return rfm