# Unit Testing - It means testing one small isolated piece of code
# It asks if i give the function this input, do i get the expected output
# For Example: Input: Ready To Move, Expected Output: 1
# Tool Use: pytest

# Mocking Concept In Tetsing:
# - We Should not rely on external dependencies like databases, APIs, or file systems during unit testing.
# - Mocking allows us to create fake versions of these dependencies, so we can test our code in isolation without actually connecting to them.
# - For Example: calling external API, download artifacts, depend on mlflow registry, depend on dagshub etc.

import pandas as pd
import numpy as np
from src.api.utils.model_loader import ModelLoader
from src.api.services.inference import PredictionService

def test_preprocess_input_unseen_location(monkeypatch):

    def mock_known_locations():
        return ['Whitefield','Electronic City Phase II']

    # monkeypatch is a fixture provided by pytest that allows us to temporarily modify 
    # or replace attributes, functions, or objects during testing.
    # we directly pass monkeypatch as an argument to the test function no need to import it, and 
    # then we can use it to replace the load_known_locations method of the ModelLoader 
    # class with our mock function that returns a predefined list of known locations.

    monkeypatch.setattr(ModelLoader, 'load_known_locations', mock_known_locations)


    input_data = {
        'area_type': 'Super built-up Area',
        'availability': 'Ready To Move',
        'location': 'Mars',
        'size': '2 BHK',
        'total_sqft': '1200',
        'bath': 2,
        'balcony': 1
    }

    processed_df = PredictionService.preprocess_input(input_data)

    assert processed_df.loc[0, 'original_location'] == 'Other'

def test_predict_returns_prediction(monkeypatch):


    class MockModel:

       feature_names_in_ = [
            'availability',
            'balcony',
            'new_total_sqft',
            'bath_per_bhk',
            'Super built-up  Area',
            'location_Whitefield',
            'location_Other'
       ]

       def predict(self, X):
            return np.array([4.5])  # Mock prediction value
    
    def mock_load_model():
        return MockModel()

    def mock_known_locations():
        return ['Whitefield','Electronic City Phase II']
    
    def mock_upload_prediction_log(**kwargs):
        pass

    monkeypatch.setattr(ModelLoader, 'load_model', mock_load_model)
    monkeypatch.setattr(ModelLoader, 'load_known_locations', mock_known_locations)
    monkeypatch.setattr('src.api.services.inference.upload_prediction_log', mock_upload_prediction_log)

    # this is how data looks in frontend application when user is trying to predict the price of a house
    input_data = {
        'area_type': 'Super built-up Area',
        'availability': 'Ready To Move',
        'location': 'Whitefield',
        'size': '2 BHK',
        'total_sqft': '1200',
        'bath': 2,
        'balcony': 1
    }

    prediction = PredictionService.predict(input_data)

    assert isinstance(prediction, np.float64) or isinstance(prediction, float)
    assert prediction > 0  # price should be positive

# run using the following commands in the terminal from the root directory of the project:
# pytest tests/test_inference.py -v