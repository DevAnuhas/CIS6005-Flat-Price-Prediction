# CIS6005 Computational Intelligence

## ITMO-HDU Flat Price Prediction 2025-2026

### Project Overview

This project implements a comprehensive machine learning pipeline for predicting flat/apartment prices using multiple models including Linear Regression, Random Forest, XGBoost, and Neural Networks. The project includes a production-ready REST API built with FastAPI.

### Competition Details

- **Competition**: ITMO-HDU Flat Price Prediction 2025-2026
- **Task**: Regression (predict flat/apartment prices)
- **Kaggle Link**: <https://www.kaggle.com/competitions/itmo-flat-price-prediction-2025-2026>

### Project Structure

```
CIS6005-Flat-Price-Prediction/
├── data/
│   ├── raw/                    # Original datasets from Kaggle
│   └── processed/              # Preprocessed datasets
├── notebooks/                  # Jupyter notebooks for exploration
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Model_Development.ipynb
│   └── 04_Model_Evaluation.ipynb
├── src/                        # Source code modules
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   └── utils.py
├── api/                        # FastAPI application
│   ├── main.py
│   ├── schemas.py
│   └── model_loader.py
├── models/                     # Saved trained models
├── reports/                    # Report and figures
│   └── figures/
├── submissions/                # Kaggle submission files
│   └── kaggle_submissions/
└── requirements.txt
```

### Installation

1. Clone the repository:

```bash
git clone https://github.com/DevAnuhas/CIS6005-Flat-Price-Prediction.git
cd CIS6005-Flat-Price-Prediction
```

1. Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

1. Install dependencies:

```bash
pip install -r requirements.txt
```

### Usage

#### 1. Download Dataset

- Navigate to: <https://www.kaggle.com/competitions/itmo-flat-price-prediction-2025-2026>
- Download `train.csv`, `test.csv`, and `sample_submission.csv`
- Place them in the `data/raw/` directory

#### 2. Run Exploratory Data Analysis

```bash
jupyter notebook notebooks/01_EDA.ipynb
```

#### 3. Train Models

```bash
jupyter notebook notebooks/03_Model_Development.ipynb
```

#### 4. Run API Server

```bash
uvicorn api.main:app --reload --port 8000
```

Access API documentation at: <http://localhost:8000/docs>

#### 5. Generate Kaggle Submission

```python
# See notebooks/04_Model_Evaluation.ipynb
```

### Dataset Features

- `kitchen_area`: Kitchen area in square meters
- `bath_area`: Bathroom area in square meters
- `other_area`: Other area in square meters
- `gas`: Gas availability (0/1)
- `hot_water`: Hot water availability (0/1)
- `central_heating`: Central heating availability (0/1)
- `extra_area`: Extra area in square meters
- `extra_area_count`: Count of extra areas
- `year`: Year built
- `ceil_height`: Ceiling height in meters
- `floor_max`: Total floors in building
- `floor`: Floor number
- `total_area`: Total area in square meters
- `bath_count`: Number of bathrooms
- `extra_area_type_name`: Type of extra area
- `district_name`: District name
- `rooms_count`: Number of rooms
- `price`: Target variable (flat price)

### Models Implemented

1. **Linear Regression** (Baseline)
2. **Random Forest** (Ensemble)
3. **XGBoost** (Gradient Boosting)
4. **Neural Network (MLP)** (Deep Learning)
5. **Stacking Ensemble** (Meta-learner)

### API Endpoints

- `POST /predict` - Predict flat price for single input
- `POST /predict/batch` - Predict prices for multiple flats
- `GET /health` - Health check
- `GET /model-info` - Model information

### Contributing

This is an academic project for CIS6005 module assessment.

### License

Academic Use Only

### Author

Name: D.M.C. Anuhas Dissanayake
Student ID: CL/BSCSD/32/122
