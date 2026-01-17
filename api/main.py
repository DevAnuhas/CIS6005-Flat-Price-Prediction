"""
FastAPI Main Application
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
import joblib
import numpy as np
import pandas as pd
import os

app = FastAPI(
    title="ITMO Flat Price Prediction API",
    description="REST API for predicting flat prices using trained ML models",
    version="1.0.0"
)

# Global variables for models
model: Any = None
scaler: Optional[Any] = None
label_encoders: Optional[Dict[str, Any]] = None
model_name: str = "Unknown"


@app.on_event("startup")
async def load_models():
    """Load models at startup."""
    global model, scaler, label_encoders, model_name

    try:
        # Load scaler and encoders (always needed)
        scaler = joblib.load('models/scaler.pkl')
        label_encoders = joblib.load('models/label_encoders.pkl')

        # Load best model name
        with open('models/best_model.txt', 'r') as f:
            model_name = f.read().strip()

        # Load the appropriate model
        if model_name == 'Neural Network':
            from tensorflow import keras  # type: ignore
            model = keras.models.load_model('models/neural_network.keras')
        elif model_name == 'XGBoost':
            model = joblib.load('models/xgboost.pkl')
        elif model_name == 'Random Forest':
            model = joblib.load('models/random_forest.pkl')
        elif model_name == 'Linear Regression':
            model = joblib.load('models/linear_regression.pkl')

        print(f"API started successfully with model: {model_name}")
    except Exception as e:
        print(f"Warning: Could not load models: {e}")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "ITMO Flat Price Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "model_info": "/model-info",
            "predict": "/predict"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """API health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.get("/model-info", tags=["Info"])
async def model_info():
    """Return information about the deployed model."""
    return {
        "model_type": "Machine Learning Ensemble",
        "framework": "Scikit-learn / TensorFlow",
        "target": "Flat Price (Currency)",
        "training_date": "2026",
        "version": "1.0.0",
        "features": [
            "kitchen_area", "bath_area", "other_area",
            "gas", "hot_water", "central_heating",
            "extra_area", "extra_area_count", "year",
            "ceil_height", "floor_max", "floor",
            "total_area", "bath_count", "rooms_count",
            "extra_area_type_name", "district_name"
        ]
    }


# Import schemas
from api.schemas import FlatFeatures, PredictionResponse


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_price(features: FlatFeatures):
    """
    Predict flat price based on input features.

    Returns predicted price and model metadata.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train and load a model first."
        )

    try:
        # Create DataFrame with input features (exclude 'other_area')
        feature_dict = {
            'kitchen_area': features.kitchen_area,
            'bath_area': features.bath_area,
            'gas': features.gas,
            'hot_water': features.hot_water,
            'central_heating': features.central_heating,
            'extra_area': features.extra_area,
            'extra_area_count': features.extra_area_count,
            'year': features.year,
            'ceil_height': features.ceil_height,
            'floor_max': features.floor_max,
            'floor': features.floor,
            'total_area': features.total_area,
            'bath_count': features.bath_count,
            'rooms_count': features.rooms_count,
            'extra_area_type_name': features.extra_area_type_name,
            'district_name': features.district_name
        }

        df = pd.DataFrame([feature_dict])

        # Feature engineering (same as notebook 02)
        # Compute living_area (was used during training): total_area - kitchen - bath
        df['living_area'] = (df['total_area'] - df['kitchen_area'] - df['bath_area']).clip(lower=0)
        df['floor_ratio'] = df['floor'] / df['floor_max']
        df['is_ground_floor'] = (df['floor'] == 1).astype(int)
        df['is_top_floor'] = (df['floor'] == df['floor_max']).astype(int)

        # Encode categorical variables
        if label_encoders is None:
            raise HTTPException(status_code=503, detail="Label encoders not loaded")

        for col in ['district_name', 'extra_area_type_name']:
            le = label_encoders.get(col)
            if le is None:
                # If encoder missing, mark unseen
                df[col] = -1
                continue

            # Handle unseen categories
            val = df[col].iloc[0]
            if val in getattr(le, 'classes_', []):
                df[col] = le.transform(df[col].astype(str))
            else:
                df[col] = -1  # Unseen category

        # Ensure DataFrame columns match features used during training (order matters)
        feature_names_path = os.path.join('data', 'processed', 'feature_names.txt')
        if os.path.exists(feature_names_path):
            with open(feature_names_path, 'r') as f:
                expected_cols = [c.strip() for c in f.read().splitlines() if c.strip()]

            # Verify all expected columns are present
            missing = [c for c in expected_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Missing required feature columns before scaling: {missing}")

            # Reorder columns to match training
            df = df[expected_cols]
        else:
            # If feature list is not available, proceed but warn
            print('Warning: feature_names.txt not found; relying on current DataFrame column order')

        # Scale features
        if scaler is None:
            raise HTTPException(status_code=503, detail="Scaler not loaded")
        
        feature_array = scaler.transform(df)

        # Make prediction
        if model_name == 'Neural Network':
            prediction = model.predict(feature_array, verbose=0).flatten()[0]
        else:
            prediction = model.predict(feature_array)[0]

        # Ensure positive prediction
        prediction = max(0, float(prediction))

        return PredictionResponse(
            predicted_price=round(prediction, 2),
            model_used=model_name,
            confidence_interval=None
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(features_list: List[FlatFeatures]):
    """Predict prices for multiple flats."""
    predictions = []
    for features in features_list:
        pred = await predict_price(features)
        predictions.append(pred)
    return {"predictions": predictions, "count": len(predictions)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
