# Orhestration pipeline for retraining the model with new data

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.training.evaluate import evaluate_production_model
from src.data.load_data import load_raw_data
from src.training.preprocessing import preprocess_for_training
from src.training.split import split_data
from src.training.train import run_training_pipeline

def retrain_model():
    
    """
    Orchestrate the retraining pipeline for the model using new data.
    
    Parameters:
        None
    Returns:
        None
    """

    r2, rmse, needs_retraining = evaluate_production_model()

    if not needs_retraining:
        print("Model performance is satisfactory. No retraining needed.")
        return
    
    # metrics
    print("Model performance has degraded. Initiating retraining pipeline...")
    print(f"Production Model R2 Score: {r2}")
    print(f"Production Model RMSE: {rmse}")

    BASE_DIR = Path(__file__).resolve().parents[2]

    train_data_path = (BASE_DIR / 
                        "data" / 
                        "processed" / 
                        "train_data.csv")
        
    new_data_path = (BASE_DIR / 
                    "data" / 
                    "processed" / 
                    "new_data.csv")

    # Load both the datasets
    # both datasets should have the same structure and columns for concatenation to work properly
    # both datasets are featured enginnered datasets ready for training
    # if the new data has different columns, we need to handle that before concatenation (e.g., by adding missing columns with default values)
    train_df = load_raw_data(train_data_path, source_type="csv")
    new_df = load_raw_data(new_data_path, source_type="csv")

    combined_df = pd.concat([train_df, new_df], ignore_index=True)
    print(f"Combined dataset shape: {combined_df.shape}")

    X,y = preprocess_for_training(combined_df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Starting retraining pipeline with combined dataset...")
    model,run_id = run_training_pipeline(X_train, X_test, y_train, y_test)

    print(f"Retraining completed. New model run ID: {run_id}")



if __name__ == "__main__":
    retrain_model()

# python src/pipelines/retrain_pipeline.py