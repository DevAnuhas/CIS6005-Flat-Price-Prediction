# CIS6005 Computational Intelligence

## ITMO-HDU Flat Price Prediction 2025-2026

## Overview

This project implements a complete machine learning pipeline for predicting flat/apartment prices using multiple models including Linear Regression, Random Forest, XGBoost, and Neural Networks. The project includes a production-ready REST API built with FastAPI.

## Project Structure

```
CIS6005-Flat-Price-Prediction/
├── data/
│   ├── raw/                        # Original Kaggle datasets
│   │   ├── data.csv                # Training data with prices
│   │   ├── test.csv                # Test data for predictions
│   │   └── solution_example.csv
│   └── processed/                  # Preprocessed datasets (generated)
├── notebooks/                      # Jupyter notebooks (core implementation)
│   ├── 01_EDA.ipynb                # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb      # Data cleaning & feature engineering
│   ├── 03_Model_Development.ipynb  # Train 4 ML models
│   └── 04_Model_Evaluation.ipynb   # Evaluation & Kaggle submission
├── api/                            # Production REST API
│   ├── main.py                     # FastAPI application
│   ├── schemas.py                  # Pydantic request/response models
│   └── model_loader.py             # Model loading utilities
├── models/                         # Saved trained models (generated)
├── submissions/                    # Kaggle submission files (generated)
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

Download the competition data from Kaggle:
<https://www.kaggle.com/competitions/itmo-flat-price-prediction-2025-2026>

Place files in `data/raw/`:

- `data.csv` (training data)
- `test.csv` (test data)

## Usage

### Run Notebooks

Execute notebooks in sequence:

```bash
jupyter notebook notebooks/01_EDA.ipynb
jupyter notebook notebooks/02_Preprocessing.ipynb
jupyter notebook notebooks/03_Model_Development.ipynb
jupyter notebook notebooks/04_Model_Evaluation.ipynb
```

Or run all at once:

```bash
jupyter notebook
```

### Run API Server

After training models (notebook 03), start the API:

```bash
python -m uvicorn api.main:app --reload --port 8000
```

Access API documentation: <http://localhost:8000/docs>

### Test API

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

## Models Implemented

1. **Linear Regression** - Baseline model
2. **Random Forest** - Ensemble of decision trees
3. **XGBoost** - Gradient boosting
4. **Neural Network** - Multi-layer perceptron (128→64→32 architecture)

## Features

### Original Features (17)

- **Numerical** (12): kitchen_area, bath_area, other_area, extra_area, extra_area_count, year, ceil_height, floor_max, floor, total_area, bath_count, rooms_count
- **Categorical** (5): gas, hot_water, central_heating, extra_area_type_name, district_name

### Engineered Features (4)

- `floor_ratio`: Floor position relative to building height
- `is_ground_floor`: Binary indicator for ground floor
- `is_top_floor`: Binary indicator for top floor
- `living_area`: Usable space excluding kitchen and bathroom

## Preprocessing Pipeline

1. **Missing Value Handling**: Median imputation (numerical), mode imputation (categorical)
2. **Outlier Capping**: IQR method (1.5 \* IQR threshold)
3. **Feature Engineering**: Create 4 derived features
4. **Categorical Encoding**: Label encoding for district and extra area type
5. **Feature Scaling**: StandardScaler normalization
6. **Train-Val Split**: 80/20 split for model validation

## Evaluation Metrics

- **RMSE**: Root Mean Squared Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)
- **R²**: Coefficient of determination (higher is better, max 1.0)
- **MAPE**: Mean Absolute Percentage Error (interpretability)

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `GET /model-info` - Model metadata
- `POST /predict` - Single flat price prediction
- `POST /predict/batch` - Batch predictions

## Dependencies

```txt
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
xgboost>=2.0.0
tensorflow>=2.15.0
jupyter>=1.0.0
joblib>=1.3.0
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.5.0
```

## Results

Results will be available after running the notebooks:

- Model comparison table: notebook 03
- Best model selection: notebook 03
- Kaggle submission: `submissions/final_submission.csv`
- Validation metrics: notebook 04

## Kaggle Submission

1. Run notebooks 01 → 02 → 03 → 04
2. Find submission file: `submissions/final_submission.csv`
3. Upload to: <https://www.kaggle.com/competitions/itmo-flat-price-prediction-2025-2026/submit>

## Future Improvements

- Hyperparameter optimization (GridSearch, Bayesian)
- Ensemble methods (stacking multiple models)
- Additional feature engineering (polynomial, interactions)
- Cross-validation for robust evaluation
- Model interpretability (SHAP values)

## License

Academic project for educational purposes.

## Author

Name: D.M.C. Anuhas Dissanayake
Student ID: CL/BSCSD/32/122
Module: CIS6005 Computational Intelligence

## Competition

ITMO-HDU Flat Price Prediction 2025-2026
<https://www.kaggle.com/competitions/itmo-flat-price-prediction-2025-2026>
