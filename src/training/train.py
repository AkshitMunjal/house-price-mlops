import mlflow
import dagshub
import mlflow.xgboost
import numpy as np
from typing import Tuple
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error,r2_score
from src.training.save_model import (log_feature_columns,
                                     save_known_locations,
                                     log_known_locations
                                    )


# dagshub integration
dagshub.init(
        repo_owner="AkshitMunjal",
        repo_name="house-price-mlops",
        mlflow=True
)


def run_training_pipeline(X_train, X_test, y_train, y_test) -> Tuple[XGBRegressor, str]:
    """
    Train an XGBoost regression model and evaluate its performance.

    Parameters:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training target variable.
        y_test (pd.Series): Testing target variable.

    Returns:
        XGBRegressor: The best XGBoost regression model found through grid search.
        run_id (str): The ID of the MLflow run.
    """
   

    y_train = np.log(y_train)
    y_test = np.log(y_test)

    model = XGBRegressor(random_state=42)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1],
        'colsample_bytree': [0.8, 1]
    }

    grid = GridSearchCV(estimator=model, 
                        param_grid=param_grid, 
                        cv=5, 
                        verbose=1,
                        scoring='neg_mean_squared_error', 
                        n_jobs=-1)



    # mlflow tracking
    with mlflow.start_run():

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
 
        # extract all valid training locations
        # from one-hot encoded location features
        known_locations = []

        for col in X_train.columns:

            # all encoded columns start with location_
            if col.startswith('location_'):

                # remove prefix to get the actual location name
                # example: location_Electronic City Phase II -> Electronic City Phase II
                location_name = col.replace('location_', '')

                known_locations.append(location_name)

        # save the known locations list to disk using the save_known_locations function from save_model.py
        save_known_locations(known_locations)

        # log the known locations list to mlflow using the log_known_locations function from save_model.py
        log_known_locations()

        y_pred = best_model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"Best Parameters: {grid.best_params_}")
        print(f"Root Mean Squared Error: {rmse}")
        print(f"R^2 Score: {r2}")

        # log parameters and metrics to mlflow
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # log the best model to mlflow
        mlflow.xgboost.log_model(best_model, "best_xgboost_model")
        
        # log the feature columns to mlflow
        log_feature_columns()

        # get mlflow run id
        run_id = mlflow.active_run().info.run_id

        # Save the trained model and feature columns in artifacts folder
        # save_model(best_model, X_train.columns)

    return best_model,run_id