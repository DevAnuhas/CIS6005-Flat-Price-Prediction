"""
Model Loading Utilities for API
"""

import joblib
import os
from tensorflow import keras  # type: ignore


class ModelLoader:
    """Handles loading and caching of trained models."""

    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.loaded_models = {}

    def load_sklearn_model(self, model_name):
        """Load scikit-learn model."""
        filepath = os.path.join(self.models_dir, f'{model_name}.pkl')

        if model_name in self.loaded_models:
            return self.loaded_models[model_name]

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model not found: {filepath}")

        model = joblib.load(filepath)
        self.loaded_models[model_name] = model
        return model

    def load_keras_model(self, model_name):
        """Load Keras/TensorFlow model."""
        filepath = os.path.join(self.models_dir, f'{model_name}.h5')

        if model_name in self.loaded_models:
            return self.loaded_models[model_name]

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model not found: {filepath}")

        model = keras.models.load_model(filepath)
        self.loaded_models[model_name] = model
        return model

    def load_preprocessor(self, preprocessor_name='preprocessor'):
        """Load preprocessing pipeline."""
        return self.load_sklearn_model(preprocessor_name)

    def load_scaler(self, scaler_name='scaler'):
        """Load feature scaler."""
        return self.load_sklearn_model(scaler_name)

    def get_model(self, model_name):
        """Get model (loads if not cached)."""
        if model_name.endswith('.h5'):
            return self.load_keras_model(model_name.replace('.h5', ''))
        return self.load_sklearn_model(model_name)
