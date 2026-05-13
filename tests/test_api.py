from fastapi.testclient import TestClient
from src.api.main_api import app

client = TestClient(app)

def test_predict_endpoint(monkeypatch):
    
    def mock_predict(input_data):
        return 95.5  # Mocked predicted price in lakhs

    monkeypatch.setattr("src.api.services.inference.PredictionService.predict", mock_predict)

    response = client.post(
        "/predict",
        json={
            "area_type": "Super built-up  Area",
            "availability": "Ready To Move",
            "location": "Electronic City Phase II",
            "size": "2 BHK",
            "total_sqft": "1200",
            "bath": 2,
            "balcony": 1
        }
    )

    assert response.status_code == 200

    data = response.json()
    assert "predicted_price_lakhs" in data
    assert data["predicted_price_lakhs"] == 95.5

# Use the following command to run the tests:
# pytest tests/test_api.py -v