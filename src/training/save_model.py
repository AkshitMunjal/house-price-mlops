import mlflow
import os
import joblib

# def save_model(model,feature_columns,save_dir="artifacts") -> None:

#     """
#     Saves the trained model and feature columns to disk.
    
#     Parameters:
#         - model: The trained machine learning model to be saved.
#         - feature_columns: List of feature column names used in training.
#         - save_dir: Directory where the model and feature columns will be saved (default: "artifacts").

#     Returns:
#         - None
    
#     """

#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
    
#     model_path = os.path.join(save_dir, 'trained_model.joblib')    
#     features_path = os.path.join(save_dir, 'feature_columns.joblib')

#     joblib.dump(model, model_path)
#     joblib.dump(feature_columns, features_path)
    
#     print(f"Model saved to {model_path}")
#     print(f"Feature columns saved to {features_path}")


def log_feature_columns() -> None:
    
    """
    Logs the feature columns used in training to MLflow from the artifacts folder.

    Parameters:
        - None
    Returns:
        - None

    """

    mlflow.log_artifact('artifacts/feature_columns.joblib', artifact_path='feature_columns')
    print("Feature columns logged to MLflow")

def save_known_locations(known_locations:list,save_dir="artifacts") -> None:

    """
    Saves the known locations list to disk.

    Parameters:
        - known_locations(list): List of known locations to be saved.
        - save_dir: Directory where the known locations will be saved (default: "artifacts").

    Returns:
        - None
    
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    locations_path = os.path.join(save_dir, 'known_locations.joblib')    

    joblib.dump(known_locations, locations_path)
    
    print(f"Known locations saved to {locations_path}")


def log_known_locations() -> None:

    """
    Logs the known locations list to MLflow from the artifacts folder.

    Parameters:
        - None
    
    Returns:
        - None
    """

    mlflow.log_artifact('artifacts/known_locations.joblib', artifact_path='known_locations')
    print("Known locations logged to MLflow")