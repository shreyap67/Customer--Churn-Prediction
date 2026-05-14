"""Generate sample_customers.csv for bulk upload demo."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset.generate_dataset import generate_churn_dataset

out = Path(__file__).parent / "sample_customers.csv"
df = generate_churn_dataset(n_samples=200, output_path=out)
# Remove Churn column (user uploads without ground truth)
df_demo = df.drop(columns=["Churn"])
df_demo.to_csv(out, index=False)
print(f"Sample data saved: {out}")
