import os
import pandas as pd
import numpy as np

np.random.seed(42)

dates = pd.date_range(start="2026-01-01", periods=100, freq="D")
revenue = 10000 + np.cumsum(np.random.normal(50, 100, 100))

revenue[50] = revenue[50] * 0.3
df = pd.DataFrame({
    "Date": dates,
    "Revenue": revenue.round(2)
})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(BASE_DIR, "data")
os.makedirs(data_dir, exist_ok=True)

output_path = os.path.join(data_dir, "business_metrics.xlsx")
df.to_excel(output_path, index=False)
print(f"Test file ban gayi: {output_path}! Din 50 pe anomaly hai.")
