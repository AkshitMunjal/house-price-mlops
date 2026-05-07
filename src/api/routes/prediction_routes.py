from fastapi import APIRouter
from src.api.schemas import HouseData, PricePredictionResponse
from src.api.services.inference import PredictionService


router = APIRouter()

@router.post("/predict", response_model=PricePredictionResponse)
def predict_price(house_data: HouseData) -> PricePredictionResponse:
    """
    Endpoint to predict the price of a house based on the input data.

    Args:
        house_data (HouseData): The input data for the house, including area type, availability, location, size, total square footage, number of bathrooms, and number of balconies.

    Returns:
        PricePredictionResponse: The predicted price in lakhs.
    """

    predicted_price = PredictionService.predict(house_data.model_dump())
    
    return PricePredictionResponse(predicted_price_lakhs=predicted_price)


# run this app using uvicorn using below command:
# uvicorn src.api.main_api:app --reload