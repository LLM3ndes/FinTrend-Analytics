import os
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def generate_ecommerce_data(
    num_customers: int = 500, num_records: int = 3000
) -> None:
  
  np.random.seed(42)
  random.seed(42)

  customer_ids = [f"CUST-{1000 + i}" for i in range(num_customers)]
  end_date = datetime.now()
  start_date = end_date - timedelta(days=365)

  data = []
  for _ in range(num_records):
    cust_id = random.choice(customer_ids)
    days_offset = random.randint(0, 365)
    trans_date = start_date + timedelta(days=days_offset)

    amount = round(float(np.random.exponential(scale=150) + 20), 2)
    quantity = random.randint(1, 5)

    data.append({
        "transaction_id": f"TX-{random.randint(100000, 999999)}",
        "customer_id": cust_id,
        "transaction_date": trans_date.strftime("%Y-%m-%d"),
        "amount": amount,
        "quantity": quantity,
    })

  df = pd.DataFrame(data)

  script_dir = os.path.dirname(os.path.abspath(__file__))
  output_path = os.path.join(script_dir, "sales_data.csv")
  df.to_csv(output_path, index=False)

  print(f"[+] Dataset gerado com sucesso em: {output_path}")
  print(
      f"[+] Total de transações: {len(df)} | Clientes únicos:"
      f" {df['customer_id'].nunique()}"
  )


if __name__ == "__main__":
  generate_ecommerce_data()