from fastapi import FastAPI
import uvicorn
import pickle
from pydantic import BaseModel
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine

# Create the FastAPI app
app = FastAPI(
    title="Wine Quality Predictor",
    description="Predict wine quality based on wine features",
    version="0.1",
)

# Create and save a model if it doesn't exist
def create_model():
    # Load the wine dataset
    wine = load_wine()
    X, y = wine.data, wine.target
    
    # Create a simple Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Save the model
    with open("models/wine_rf_model.pkl", 'wb') as file:
        pickle.dump(model, file)
    
    return model, wine.feature_names

# Define the request body model
class WineRequest(BaseModel):
    features: list[float]

# Global variables
model = None
feature_names = None

# Load the model on startup
@app.on_event('startup')
def load_model():
    global model, feature_names
    model_path = "models/wine_rf_model.pkl"
    
    # Create model if it doesn't exist
    if not os.path.exists(model_path):
        model, feature_names = create_model()
        print("Model created and saved successfully!")
    else:
        # Load existing model
        try:
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
            # Use wine dataset feature names
            _, feature_names = load_wine(return_X_y=False, as_frame=True)
            feature_names = feature_names.columns.tolist()
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            model, feature_names = create_model()
            print("Created new model due to loading error!")

# Define the root endpoint
@app.get('/')
def root():
    return {"message": "Wine Quality Predictor API", "feature_names": feature_names}

# Define the prediction endpoint
@app.post('/predict')
def predict(request: WineRequest):
    # Convert the request to a numpy array
    features = np.array([request.features])
    
    # Make prediction
    prediction = model.predict(features)
    
    # Get probabilities if available
    probabilities = model.predict_proba(features)[0].tolist()
    
    # Return the prediction
    return {
        "prediction": int(prediction[0]),
        "class_probabilities": {f"class_{i}": prob for i, prob in enumerate(probabilities)},
        "feature_names": feature_names,
        "input_features": request.features
    }

# Run the app if this file is executed directly
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)