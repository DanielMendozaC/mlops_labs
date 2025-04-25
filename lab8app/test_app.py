import requests
import json

def test_prediction():
    # Define the API URL
    url = "http://localhost:8000/predict"
    
    # Example wine features
    wine_features = {
        "features": [13.2, 1.78, 2.14, 11.2, 100, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.40, 1050]
    }
    
    # First, check what features are needed by querying the root endpoint
    feature_response = requests.get("http://localhost:8000/predict")
    if feature_response.status_code == 200:
        feature_data = feature_response.json()
        feature_names = feature_data.get("feature_names", [])
        if feature_names:
            print(f"Available features: {feature_names}")
            print(f"Number of features: {len(feature_names)}")
    
    # Make the POST request
    response = requests.post(url, json=wine_features)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Get the prediction result
        result = response.json()
        
        # Print the result
        print("\nPrediction Results:")
        print("Predicted Wine Class:", result["prediction"])
        print("\nClass Probabilities:")
        for class_name, probability in result["class_probabilities"].items():
            print(f"  {class_name}: {probability:.4f}")
        
        return result
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

if __name__ == "__main__":
    result = test_prediction()
    print("\nRaw Response:")
    print(json.dumps(result, indent=4))