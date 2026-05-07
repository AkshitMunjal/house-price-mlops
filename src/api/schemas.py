from pydantic import BaseModel, Field


class HouseData(BaseModel):
    area_type: str = Field(..., example="Super built-up  Area")
    availability: str = Field(..., example="Ready To Move")
    location: str = Field(..., example="Electronic City Phase II")
    size: str = Field(..., example="1 BHK")
    total_sqft: str = Field(..., example="1500-1700")
    bath: float = Field(..., gt=0, le=10, example=2)
    balcony: float = Field(..., ge=0, le=5, example=1)


class PricePredictionResponse(BaseModel):
    predicted_price_lakhs: float = Field(..., example=125.22)