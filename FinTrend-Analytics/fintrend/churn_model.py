import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split


def prepare_churn_features(
    df: pd.DataFrame, rfm_df: pd.DataFrame, churn_threshold_days: int = 90
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
  """Prepara a matriz de atributos (X) e a variável alvo (y: Churn = 1 se recência > churn_threshold_days)."""
  features = rfm_df.copy()
  # Rótulo de Churn: 1 se inativo há mais de X dias, senão 0
  features["is_churn"] = (features["recency"] > churn_threshold_days).astype(int)

  # Engenharia de Atributos: Ticket Médio e Quantidade Média por pedido
  df_clean = df.copy()
  avg_metrics = (
      df_clean.groupby("customer_id")
      .agg({"amount": "mean", "quantity": "mean"})
      .reset_index()
  )
  avg_metrics.columns = ["customer_id", "avg_ticket", "avg_quantity"]
  features = pd.merge(features, avg_metrics, on="customer_id", how="left")

  X = features[[
      "recency",
      "frequency",
      "monetary",
      "avg_ticket",
      "avg_quantity",
  ]]
  y = features["is_churn"]
  return X, y, features


def train_churn_model(
    X: pd.DataFrame, y: pd.Series
) -> tuple[
    RandomForestClassifier, dict, float, pd.DataFrame
]:  # noqa: E501
  """Treina o algoritmo Random Forest para calcular a probabilidade de Churn de cada cliente."""
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.25, random_state=42, stratify=y
  )
  model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
  model.fit(X_train, y_train)

  # Avaliação do Modelo
  y_pred = model.predict(X_test)
  y_proba = model.predict_proba(X_test)[:, 1]
  auc_score = roc_auc_score(y_test, y_proba)
  report = classification_report(y_test, y_pred, output_dict=True)

  # Importância das variáveis para explicabilidade (Explainable AI)
  feature_importance = pd.DataFrame({
      "feature": X.columns,
      "importance": model.feature_importances_,
  }).sort_values("importance", ascending=False)

  return model, report, auc_score, feature_importance