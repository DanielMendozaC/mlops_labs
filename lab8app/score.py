import joblib
import numpy as np

MODEL_PATH = "models/wine_rf_model.pkl"  # or mlflow_wine_model.pkl
clf = joblib.load(MODEL_PATH)

def predict(features: np.ndarray) -> np.ndarray:
    return clf.predict(features)
