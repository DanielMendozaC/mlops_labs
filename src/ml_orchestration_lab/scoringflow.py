from metaflow import FlowSpec, step, Parameter, Flow
import pandas as pd
import numpy as np
import mlflow
import os

class ScoringFlow(FlowSpec):
    # Parameters
    model_name = Parameter('model_name', default='wine-classifier', help='Name of the registered model')
    model_version = Parameter('model_version', default=None, help='Version of the model to use', type=str)
    
    @step
    def start(self):
        """
        Start step: Load data to score and set up
        """
        print("Loading data to score...")

        # Load data
        train_run = Flow('TrainingFlow').latest_run
        self.X_test = train_run.data.X_test
        self.y_test = train_run.data.y_test
        self.feature_names = train_run.data.feature_names
        self.target_names = train_run.data.target_names
        
        print(f"Data loaded: {self.X_test.shape[0]} samples")
        
        self.next(self.load_model)
    
    @step
    def load_model(self):
        """
        Load the trained model from MLFlow
        """
        print(f"Loading model: {self.model_name}")
        
        # Set up MLFlow
        mlflow.set_tracking_uri('http://localhost:5001')  
        
        # Load the model
        if self.model_version:
            model_uri = f"models:/{self.model_name}/{self.model_version}"
        else:
            model_uri = f"models:/{self.model_name}/latest"
        
        self.model = mlflow.sklearn.load_model(model_uri)
        
        print(f"Model loaded: {self.model}")
        
        self.next(self.predict)
    
    @step
    def predict(self):
        """
        Generate predictions using the loaded model
        """
        print("Generating predictions...")
        
        # Make predictions
        self.predictions = self.model.predict(self.X_test)
        
        # Calculate accuracy (since we have labels)
        self.accuracy = (self.predictions == self.y_test).mean()
        
        print(f"Predictions generated for {len(self.predictions)} samples")
        print(f"Accuracy: {self.accuracy:.4f}")
        
        self.next(self.end)
    
    @step
    def end(self):
        """
        Final step: Show results
        """
        print("Scoring completed!")
        
        # Create a dataframe with results
        results = pd.DataFrame({
            'True Label': [self.target_names[i] for i in self.y_test],
            'Predicted Label': [self.target_names[i] for i in self.predictions]
        })
        
        print("\nSample predictions:")
        print(results.head(10))
        
        # Print summary
        print(f"\nAccuracy: {self.accuracy:.4f}")

if __name__ == "__main__":
    ScoringFlow()