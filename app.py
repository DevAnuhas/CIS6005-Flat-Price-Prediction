import streamlit as st
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Flat Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: 2rem;
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #023EA4 0%, #04A1FC 100%);
        color: white;
        text-align: center;
        font-size: 1.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    try:
        # Load best model name
        model_path = Path('models/best_model.txt')
        if not model_path.exists():
            st.error("Model configuration not found. Please train models first using the notebooks.")
            return None, None
        
        with open(model_path, 'r') as f:
            model_name = f.read().strip()
        
        # Load the appropriate model
        if model_name == 'Neural Network':
            from tensorflow import keras
            model = keras.models.load_model('models/neural_network.keras')
        elif model_name == 'XGBoost':
            model = joblib.load('models/xgboost.pkl')
        elif model_name == 'Random Forest':
            model = joblib.load('models/random_forest.pkl')
        elif model_name == 'Linear Regression':
            model = joblib.load('models/linear_regression.pkl')
        else:
            st.error(f"Unknown model type: {model_name}")
            return None, None
        
        return model, model_name
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None


def create_feature_array(inputs):
    # Calculate derived features
    floor_ratio = inputs['floor'] / inputs['floor_max'] if inputs['floor_max'] > 0 else 0
    is_ground_floor = 1 if inputs['floor'] == 1 else 0
    is_top_floor = 1 if inputs['floor'] == inputs['floor_max'] else 0
    living_area = inputs['total_area'] - inputs['kitchen_area'] - inputs['bath_area']
    
    # Encode categorical features
    extra_area_mapping = {
        'None': 0, 'Balcony': 1, 'Loggia': 2, 'Terrace': 3, 'Other': 4
    }
    district_mapping = {
        'Center': 0, 'North': 1, 'South': 2, 'East': 3, 'West': 4,
        'Northeast': 5, 'Northwest': 6, 'Southeast': 7, 'Southwest': 8, 'Other': 9
    }
    
    extra_area_encoded = extra_area_mapping.get(inputs['extra_area_type'], 0)
    district_encoded = district_mapping.get(inputs['district'], 0)
    
    # Create feature array in the correct order
    features = np.array([[
        inputs['kitchen_area'],
        inputs['bath_area'],
        inputs['gas'],
        inputs['hot_water'],
        inputs['central_heating'],
        inputs['extra_area'],
        inputs['extra_area_count'],
        inputs['year'],
        inputs['ceil_height'],
        inputs['floor_max'],
        inputs['floor'],
        inputs['total_area'],
        inputs['bath_count'],
        extra_area_encoded,
        district_encoded,
        inputs['rooms_count'],
        floor_ratio,
        is_ground_floor,
        is_top_floor,
        living_area
    ]])
    
    return features

def main():
    # Initialize session state for predictions
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    
    # Header
    st.markdown('<h1 class="main-header">🏠 Flat Price Prediction System</h1>', unsafe_allow_html=True)
    
    # Load model
    model, model_name = load_model()
    
    if model is None:
        st.warning("⚠️ Please train the models first by running the notebooks in sequence (01-04).")
        return
    
    # Display model info in sidebar
    with st.sidebar:
        st.header("Model Information")
        st.success(f"**Active Model:** {model_name}")
        st.info("This model was trained on ITMO-HDU flat price dataset")
        
        st.markdown("---")
    
    # Main content area
    st.markdown("### Enter Flat Details")
    
    # Create input form with columns
    col1, col2, col3 = st.columns(3)
        
    with col1:
            st.subheader("Area Information")
            total_area = st.number_input("Total Area (m²)", min_value=10.0, max_value=500.0, value=65.0, step=0.5)
            kitchen_area = st.number_input("Kitchen Area (m²)", min_value=1.0, max_value=50.0, value=10.5, step=0.5)
            bath_area = st.number_input("Bathroom Area (m²)", min_value=1.0, max_value=20.0, value=4.0, step=0.5)
            extra_area = st.number_input("Extra Area (m²)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
            extra_area_count = st.number_input("Extra Area Count", min_value=0, max_value=10, value=1)
            extra_area_type = st.selectbox("Extra Area Type", 
                                          ['None', 'Balcony', 'Loggia', 'Terrace', 'Other'])
    
    with col2:
        st.subheader("Building Details")
        year = st.number_input("Year Built", min_value=1900, max_value=2030, value=2010, step=1)
        floor_max = st.number_input("Total Floors", min_value=1, max_value=50, value=10, step=1)
        floor = st.number_input("Floor Number", min_value=1, max_value=floor_max, value=5, step=1)
        ceil_height = st.number_input("Ceiling Height (m)", min_value=2.0, max_value=5.0, value=2.7, step=0.1)
    
    with col3:
        st.subheader("Flat Configuration")
        rooms_count = st.number_input("Number of Rooms", min_value=1, max_value=10, value=2, step=1)
        bath_count = st.number_input("Number of Bathrooms", min_value=1, max_value=5, value=1, step=1)
        district = st.selectbox("District", 
                                ['Center', 'North', 'South', 'East', 'West', 
                                'Northeast', 'Northwest', 'Southeast', 'Southwest', 'Other'])
        
        st.subheader("Amenities")
        gas = st.checkbox("Gas", value=True)
        hot_water = st.checkbox("Hot Water", value=True)
        central_heating = st.checkbox("Central Heating", value=True)
    
    # Prepare inputs for prediction
    inputs = {
        'kitchen_area': kitchen_area,
        'bath_area': bath_area,
        'total_area': total_area,
        'gas': 1 if gas else 0,
        'hot_water': 1 if hot_water else 0,
        'central_heating': 1 if central_heating else 0,
        'extra_area': extra_area,
        'extra_area_count': extra_area_count,
        'year': year,
        'ceil_height': ceil_height,
        'floor_max': floor_max,
        'floor': floor,
        'bath_count': bath_count,
        'extra_area_type': extra_area_type,
        'district': district,
        'rooms_count': rooms_count
    }

    # Create feature array
    features = create_feature_array(inputs)

    # Make prediction and display in sidebar
    with st.sidebar:
        predict_btn = st.button("Predict Price", type="primary", use_container_width=True)
        
        if predict_btn:
            with st.spinner('Predicting...'):
                try:
                    prediction = model.predict(features)
                    predicted_price = prediction[0][0] if len(prediction.shape) > 1 else prediction[0]
                    
                    # Store prediction in session state
                    st.session_state.prediction_result = {
                        'price': predicted_price,
                        'total_area': total_area
                    }

                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
                    st.session_state.prediction_result = None
        
        if st.session_state.prediction_result is not None:
            result = st.session_state.prediction_result
            st.markdown(f"""
                <div class="prediction-box">
                    <div>
                        Predicted Price: <br>
                        <strong>{result['price']:,.2f} ₽</strong>
                    </div>
                   <div>
                        Price per m²: <br>
                        <strong>{result['price']/result['total_area']:,.2f} ₽</strong>
                    </div>
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
