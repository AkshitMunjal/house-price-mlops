from src.data.load_data import load_raw_data
from pathlib import Path
from src.features.preprocess import preprocess_data
from src.features.build_features import build_features
from src.training.preprocessing import preprocess_for_training
from src.training.split import split_data
from src.training.train import run_training_pipeline

BASE_DIR = Path(__file__).resolve().parent
data_path = BASE_DIR / 'data' / 'processed' / 'train_data.csv'

df = load_raw_data(data_path, source_type='csv')
print("Original shape:", df.shape)

df_preprocessed = preprocess_data(df)
print("Preprocessed shape:", df_preprocessed.shape)

df_features = build_features(df_preprocessed)
print("Feature Engineering shape:", df_features.shape)

df_final = preprocess_for_training(df_features)
print("Final shape:", df_final[0].shape, df_final[1].shape)

X_train, X_test, y_train, y_test = split_data(df_final[0], df_final[1])
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)

model, run_id = run_training_pipeline(X_train, X_test, y_train, y_test)

print(f"Best model: {model}")
print(f"Training completed. MLflow Run ID: {run_id}")

# python main.py