import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))

import pandas as pd
import numpy as np
from src.api.utils.model_loader import ModelLoader
from src.api.utils.s3_logger import upload_prediction_log
from src.features.build_features import (
    transform_total_sqft,
    transform_size_column
)
from src.training.preprocessing import (
    encode_categorical_features,
    select_features
)
from src.api.utils.monitoring import generate_monitoring_flags


class PredictionService:
    
    # using @staticmethod decorator to define a static method that can be 
    # called on the class itself without needing an instance of the class
    # for example without this: we use obj = PredictionService(), obj.preprocess_input(input_data) 
    # but with @staticmethod we can directly call PredictionService.preprocess_input(input_data) 
    # to preprocess the input data without creating an instance of the class
    @staticmethod
    def preprocess_input(input_data: dict) -> pd.DataFrame:

        """"
        Preprocess incoming API request data
        to match training pipeline transformations.
            - Convert total_sqft to numeric
            - Encode categorical features
            - Select only the features used for prediction
        For example if the input data is:
            {"area_type": "Super built-up  Area",
            "availability": "Ready To Move",
            "location": "Electronic City Phase II",
            "size": "1 BHK",
            "total_sqft": "1500-1700",
            "bath": 2,
            "balcony": 1}
        Args:
            input_data (dict): The raw input data from the API request.
        Returns:
            pd.DataFrame: The preprocessed data ready for prediction.
        """

        # convert the input data dictionary to a pandas DataFrame
        df = pd.DataFrame([input_data])

        # availability conversion
        df['availability'] = df['availability'].apply(
            lambda x: 1 if x == 'Ready To Move' or x == 'Immediate Possession' else 0
        )

        df = transform_total_sqft(df)
        df = transform_size_column(df)

        # handle rare locations via mlflow artifacts
        known_locations = (
            ModelLoader.load_known_locations()
        )

        # convert unseen location into "Other" to match training data distribution
        if (df.loc[0,'location'] not in known_locations):
            df.loc[0,'location'] = 'Other'

        # encode categorical features using the function from preprocessing.py
        df = encode_categorical_features(df)
        
        # we have to include this because select_features() function is 
        # expecting this column to drop it otherwise it will throw an error
        df['sqft_per_bhk'] = df['new_total_sqft'] / df['new_size']
        

        # IMPORTANT:
        # we are NOT dropping features here anymore using select_features() function from preprocessing.py
        # because monitoring needs access to engineered features
        # like sqft_per_bhk, new_size, bath, etc.
        # feature selection will happen later
        # inside prepare_model_features()

        # select only the features that are used for prediction using the function from preprocessing.py
        # df = select_features(df)
        
        # this gives us full engineered dataframe with all features, 
        # including engineered features like sqft_per_bhk, new_size, bath, etc.
        # which were getting dropped before in select_features() function
        # we need these features for monitoring and explainability purposes
        return df

    @staticmethod
    def prepare_model_features(input_data: pd.DataFrame) -> pd.DataFrame:


        """
        Prepare the input data with only the features that the model was trained on.
        This is necessary because the model was trained on a specific set of features, and we need to ensure that the input data has the same features in the same order before making a prediction.
        Parameters:
            input_data (pd.DataFrame): The preprocessed input data.
        Returns:
            pd.DataFrame: The input data with only the features that the model was trained on, ready for prediction.

        """

        # drop unnecessary features that are not used for prediction
        df = select_features(input_data)

        return df

    @staticmethod
    def alignfeatures(input_data: pd.DataFrame, model) -> pd.DataFrame:

        """
        Align the input data features with the model's expected features.
        This is necessary because the model was trained on a specific set of features, and we need to ensure that the input data has the same features in the same order.

        Args:
            input_data (pd.DataFrame): The preprocessed input data.
            model: The loaded machine learning model.

        Returns:
            pd.DataFrame: The input data aligned with the model's expected features.
        """

        # get the feature names that the model was trained on
        model_features = model.feature_names_in_

        # align the input data with the model's expected features
        input_df = input_data.reindex(columns=model_features, fill_value=0)

        return input_df
    
    @staticmethod
    def predict(input_data: dict) -> float:

        """
        Make a price prediction based on the input data.
        This method preprocesses the input data, aligns it with the model's expected features, and then uses the model to make a prediction.

        Args:
            input_data (dict): The raw input data from the API request.

        Returns:
            float: The predicted price in lakhs.
        """

        # load the model using the ModelLoader class
        model = ModelLoader.load_model()

        # preprocess the input data using the preprocess_input method
        preprocessed_data = PredictionService.preprocess_input(input_data)

        # generate monitoring flags based on the preprocessed data
        monitoring_flags = generate_monitoring_flags(preprocessed_data)

        # prepare the preprocessed data with only the features that the model was trained on using the prepare_model_features method
        model_ready_data = PredictionService.prepare_model_features(preprocessed_data)

        # align the preprocessed data with the model's expected features using the alignfeatures method
        aligned_data = PredictionService.alignfeatures(model_ready_data, model)

        # make a prediction using the model's predict method and return the predicted price
        predicted_price = model.predict(aligned_data)[0]

        # reverse the log tranformation to get the actual price in lakhs
        predicted_price = np.exp(predicted_price)

        # upload prediction log to S3 using the upload_prediction_log function from s3_logger.py
        upload_prediction_log(
            input_data=input_data,
            prediction=float(predicted_price),
            monitoring_flags=monitoring_flags
        )

        
        return predicted_price
    
if __name__ == "__main__":
    # example input data for testing the prediction service
    input_data = {
        "area_type": "Super built-up  Area",
        "availability": "Ready To Move",
        "location": "Electronic City Phase II",
        "size": "1 BHK",
        "total_sqft": "1500-1700",
        "bath": 2,
        "balcony": 1
    }

    # make a prediction using the PredictionService class
    predicted_price = PredictionService.predict(input_data)
    print(f"Predicted price in lakhs: {predicted_price:.2f}")


# python src/api/services/inference.py