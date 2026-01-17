"""
Pydantic Schemas for API Request/Response
"""

from pydantic import BaseModel, Field, validator
from typing import Optional
import numpy as np


class FlatFeatures(BaseModel):
    """Input schema for flat prediction request."""

    kitchen_area: float = Field(..., ge=0, description="Kitchen area in sq meters")
    bath_area: float = Field(..., ge=0, description="Bathroom area in sq meters")
    other_area: float = Field(..., ge=0, description="Other area in sq meters")
    gas: int = Field(..., ge=0, le=1, description="Gas availability (0/1)")
    hot_water: int = Field(..., ge=0, le=1, description="Hot water availability (0/1)")
    central_heating: int = Field(..., ge=0, le=1, description="Central heating (0/1)")
    extra_area: float = Field(..., ge=0, description="Extra area in sq meters")
    extra_area_count: int = Field(..., ge=0, description="Count of extra areas")
    year: int = Field(..., ge=1900, le=2030, description="Year built")
    ceil_height: float = Field(..., ge=2, le=10, description="Ceiling height in meters")
    floor_max: int = Field(..., ge=1, description="Total floors in building")
    floor: int = Field(..., ge=1, description="Floor number")
    total_area: float = Field(..., ge=0, description="Total area in sq meters")
    bath_count: int = Field(..., ge=0, description="Number of bathrooms")
    extra_area_type_name: str = Field(..., description="Type of extra area")
    district_name: str = Field(..., description="District name")
    rooms_count: int = Field(..., ge=1, description="Number of rooms")

    @validator('floor')
    def floor_not_greater_than_max(cls, v, values):
        if 'floor_max' in values and v > values['floor_max']:
            raise ValueError('Floor cannot be greater than floor_max')
        return v

    def to_array(self):
        """Convert to numpy array for prediction."""
        return np.array([[
            self.kitchen_area,
            self.bath_area,
            self.other_area,
            self.gas,
            self.hot_water,
            self.central_heating,
            self.extra_area,
            self.extra_area_count,
            self.year,
            self.ceil_height,
            self.floor_max,
            self.floor,
            self.total_area,
            self.bath_count,
            self.rooms_count
            # Note: categorical features need encoding
        ]])

    class Config:
        schema_extra = {
            "example": {
                "kitchen_area": 10.5,
                "bath_area": 4.0,
                "other_area": 2.0,
                "gas": 1,
                "hot_water": 1,
                "central_heating": 1,
                "extra_area": 5.0,
                "extra_area_count": 1,
                "year": 2010,
                "ceil_height": 2.7,
                "floor_max": 10,
                "floor": 5,
                "total_area": 65.0,
                "bath_count": 1,
                "extra_area_type_name": "balcony",
                "district_name": "Central",
                "rooms_count": 2
            }
        }


class PredictionResponse(BaseModel):
    """Output schema for prediction response."""

    predicted_price: float = Field(..., description="Predicted flat price")
    confidence_interval: Optional[dict] = Field(None, description="95% CI if available")
    model_used: str = Field(..., description="Model used for prediction")

    class Config:
        schema_extra = {
            "example": {
                "predicted_price": 4500000.00,
                "model_used": "Neural Network (MLP)",
                "confidence_interval": {
                    "lower": 4300000.00,
                    "upper": 4700000.00
                }
            }
        }
