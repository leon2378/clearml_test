# train.py
from clearml import Task, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

# Initialize ClearML task
task = Task.init(project_name="Toy Project", task_name="Train Toy Dataset")

# Get dataset
dataset = Dataset.get(dataset_name="Toy Classification", dataset_project="Toy Project")
dataset_path = dataset.get_local_copy()
csv_file = os.path.join(dataset_path, "toy_dataset.csv")
df = pd.read_csv(csv_file)

# Train/test split
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Simple model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

print(f"Train accuracy: {train_acc}, Test accuracy: {test_acc}")

# Log metrics to ClearML
logger = task.get_logger()
logger.report_scalar("accuracy", "train", train_acc)
logger.report_scalar("accuracy", "test", test_acc)

# Save model
model_path = "toy_model.pkl"
import joblib
joblib.dump(model, model_path)
task.upload_artifact("model", model_path)