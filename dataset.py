# create_dataset.py
from clearml import Dataset
import pandas as pd
from sklearn.datasets import make_classification
import os

# Make a simple dataset
X, y = make_classification(n_samples=100, n_features=4, n_informative=2, n_classes=2, random_state=42)
df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(4)])
df["target"] = y

# Save to CSV
os.makedirs("data", exist_ok=True)
csv_path = os.path.join("data", "toy_dataset.csv")
df.to_csv(csv_path, index=False)

# Upload to ClearML
dataset = Dataset.create(
    dataset_name="Toy Classification",
    dataset_project="Toy Project",
    dataset_tags=["toy", "simple"]
)
dataset.add_files("data")  # add folder
dataset.upload()
dataset.finalize()
print("Dataset uploaded to ClearML")