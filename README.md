# CIS6005 Computational Intelligence

## ITMO-HDU Flat Price Prediction 2025-2026

## Overview

This project implements a complete machine learning pipeline for predicting flat/apartment prices using multiple models including Linear Regression, Random Forest, XGBoost, and Neural Networks. The project includes a user-friendly web interface built with Streamlit for making predictions.

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
├── models/                         # Saved trained models (generated)
├── submissions/                    # Kaggle submission files (generated)
├── app.py                          # Streamlit web application
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

### 1. Train Models

Execute notebooks in sequence to prepare data and train models:

```bash
jupyter notebook
```

Run notebooks in order:

1. `01_EDA.ipynb` - Explore the data
2. `02_Preprocessing.ipynb` - Clean and prepare features
3. `03_Model_Development.ipynb` - Train all models
4. `04_Model_Evaluation.ipynb` - Evaluate and generate Kaggle submission

### 2. Launch Streamlit App

After training models, start the web interface:

```bash
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

### 3. Make Predictions

1. Enter flat details in the web form
2. Click "Predict Price" to get instant predictions
3. View the predicted price and price per square meter

The app automatically loads the best performing model from your training session.

## Models Implemented

1. **Random Forest** - Ensemble of decision trees
2. **XGBoost** - Gradient boosting
3. **Neural Network** - Multi-layer perceptron (128→64→32 architecture)

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

## Web Application Features

- **Interactive UI**: User-friendly interface for entering flat details
- **Instant Predictions**: Real-time price predictions using trained models
- **Multiple Views**: Prediction interface and project documentation
- **Modern Design**: Clean, responsive interface with gradient styling
- **Metrics Display**: Shows predicted price, price per m², and key stats
- **Automatic Model Loading**: Uses the best performing model automatically

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
streamlit>=1.28.0
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
