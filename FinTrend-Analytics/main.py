import os
import sys
import numpy as np
import pandas as pd

# Garante que as subpastas e módulos locais sejam localizados no Windows
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
  sys.path.insert(0, BASE_DIR)

from data.generate_dataset import generate_ecommerce_data

# ATENÇÃO: As importações precisam ter o 'fintrend.' antes do nome do módulo!
from fintrend.churn_model import prepare_churn_features, train_churn_model
from fintrend.cohort import calculate_cohort_matrix
from fintrend.rfm import (
    calculate_rfm_metrics,
    calculate_rfm_scores,
    segment_customers,
)
