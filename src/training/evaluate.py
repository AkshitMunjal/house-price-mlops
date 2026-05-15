# In Evaluation we will do the following steps in order:
# 1. Load the production model that we have saved in the previous step
# 2. Load the new_data.csv file that contains the new data for evaluation
# 3. Run predictions on the new data using the loaded model
# 4. Calculate the R² score to evaluate the model's performance on the new data
# 5. Check if the model has degraded by comparing the R² score with a predefined threshold or with the R² score from the training phase

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
from sklearn.metrics import r2_score,mean_squared_error

from src.data.load_data import load_raw_data
from src.training.preprocessing import preprocess_for_training
from src.api.utils.model_loader import ModelLoader


BASE_DIR = Path(__file__).resolve().parents[2]

def evaluate_production_model():
    
    """
    Evaluate the production model on new unseen data and check for performance degradation.
    
    Parameters:
        None

    Returns:
        r2 (float): R² score of the model on the new data
        rmse (float): Root Mean Squared Error of the model on the new data
        needs_retraining (bool): Flag indicating if the model needs retraining based on performance degradation
    """

    new_data_path = (BASE_DIR / 
                    "data" / 
                    "processed" / 
                    "new_data.csv")

    threshold_r2 = 0.83

    df = load_raw_data(new_data_path, source_type="csv")
    X, y = preprocess_for_training(df)
    model = ModelLoader.load_model()

    X = X.reindex(columns=model.feature_names_in_, fill_value=0)
    y_pred = model.predict(X)
    y_pred = np.exp(y_pred)

    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    needs_retraining = r2 < threshold_r2

    return r2, rmse, needs_retraining

if __name__ == "__main__":
    
    r2, rmse, needs_retraining = evaluate_production_model()
    print("Results on unseen data:")
    print(f"R² Score: {r2}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"Needs Retraining: {needs_retraining}")



# python src/training/evaluate.py
# Results on unseen data:
# R² Score: 0.81
# Root Mean Squared Error: 32.28