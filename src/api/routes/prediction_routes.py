from fastapi import APIRouter
from src.api.schemas import HouseData, PricePredictionResponse

router = APIRouter()


@router.post("/predict", response_model=PricePredictionResponse)
def predict_price(house_data: HouseData) -> PricePredictionResponse:
    """
    Endpoint to predict house price.

    Args:
        house_data (HouseData): Input house data.

    Returns:
        PricePredictionResponse: Predicted price in lakhs.
    """

    # lazy import to avoid loading model during app startup
    from src.api.services.inference import PredictionService

    predicted_price = PredictionService.predict(
        house_data.model_dump()
    )

    return PricePredictionResponse(
        predicted_price_lakhs=predicted_price
    )


# run app:
# uvicorn src.api.main_api:app --reload