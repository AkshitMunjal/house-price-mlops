import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.load_data import load_raw_data
from src.training.preprocessing import preprocess_for_training
from src.training.split import split_data
from src.training.train import run_training_pipeline

BASE_DIR = Path(__file__).resolve().parents[2]
train_data_path = BASE_DIR / 'data' / 'processed' / 'train_data.csv'

df = load_raw_data(train_data_path, source_type='csv')
X,y = preprocess_for_training(df)
X_train, X_test, y_train, y_test = split_data(X, y)
model, run_id = run_training_pipeline(X_train, X_test, y_train, y_test)

print(f"Best model: {model}")
print(f"Training completed. MLflow Run ID: {run_id}")