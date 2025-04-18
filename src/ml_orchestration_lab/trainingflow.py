from metaflow import FlowSpec, step, Parameter
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import mlflow
import os

class TrainingFlow(FlowSpec):
    # Parameters for our flow
    random_state = Parameter('random_state', default=42, help='Random seed', type=int)
    test_size = Parameter('test_size', default=0.2, help='Test set size', type=float)
    
    @step
    def start(self):
        """
        Load and prepare the data
        """
        print("Loading data...")
        
        # For demonstration, let's use the wine dataset
        from sklearn import datasets
        
        # Load data - fixing the data access issue
        wine = datasets.load_wine()
        X = pd.DataFrame(wine['data'], columns=wine['feature_names'])
        y = wine['target']
        
        # Store data as artifacts
        self.X = X
        self.y = y
        self.target_names = wine['target_names']
        
        print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        self.next(self.split_data)
    
    @step
    def split_data(self):
        """
        Split data into train and test sets
        """
        print("Splitting data...")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        print(f"Train set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        
        # Continue to the training steps in parallel
        self.next(self.train_random_forest, self.train_logistic_regression)
    
    @step
    def train_random_forest(self):
        """
        Train a Random Forest model
        """
        print("Training Random Forest...")
        
        # Train model
        model = RandomForestClassifier(random_state=self.random_state)
        model.fit(self.X_train, self.y_train)
        
        # Evaluate model
        self.model_name = "RandomForest"
        self.model = model
        self.accuracy = model.score(self.X_test, self.y_test)
        
        print(f"Random Forest accuracy: {self.accuracy:.4f}")
        
        self.next(self.choose_model)
    
    @step
    def train_logistic_regression(self):
        """
        Train a Logistic Regression model
        """
        print("Training Logistic Regression...")
        
        # Train model
        model = LogisticRegression(random_state=self.random_state, max_iter=200)
        model.fit(self.X_train, self.y_train)
        
        # Evaluate model
        self.model_name = "LogisticRegression"
        self.model = model
        self.accuracy = model.score(self.X_test, self.y_test)
        
        print(f"Logistic Regression accuracy: {self.accuracy:.4f}")
        
        self.next(self.choose_model)
    
    @step
    def choose_model(self, inputs):
        """
        Select the best model based on performance
        """
        print("Selecting best model...")
        
        # Retrieve models and scores
        models = [(inp.model_name, inp.model, inp.accuracy) for inp in inputs]
        
        # Find the best model
        self.best_model_name, self.best_model, self.best_accuracy = max(models, key=lambda x: x[2])
        
        # Store all models and their performances for reference
        self.all_models = models
        
        # We need to explicitly handle the conflicting artifacts before merging
        # Just set these to the values from the best model
        self.model_name = self.best_model_name
        self.model = self.best_model
        self.accuracy = self.best_accuracy
        
        # Now we can merge the remaining artifacts
        self.merge_artifacts(inputs)
        
        print(f"Best model: {self.best_model_name} with accuracy: {self.best_accuracy:.4f}")
        
        self.next(self.register_model)
        
    @step
    def register_model(self):
        """
        Register the model with MLFlow
        """
        print("Registering model with MLFlow...")
        
        # Set up MLFlow
        mlflow.set_tracking_uri('http://localhost:5001')  # Adjust if your MLFlow server is elsewhere
        mlflow.set_experiment('metaflow-wine-classification')
        
        # Start a new run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("model_type", self.best_model_name)
            mlflow.log_param("random_state", self.random_state)
            mlflow.log_param("test_size", self.test_size)
            
            # Log metrics
            mlflow.log_metric("accuracy", self.best_accuracy)
            
            # Log the model
            mlflow.sklearn.log_model(
                self.best_model, 
                artifact_path="model",
                registered_model_name="wine-classifier"
            )
            
            print(f"Model registered: wine-classifier")
        
        self.next(self.end)
    
    @step
    def end(self):
        """
        Final step
        """
        print("Flow completed!")
        print(f"Best model: {self.best_model_name}")
        print(f"Accuracy: {self.best_accuracy:.4f}")
        
        # Save some info for the scoring flow
        self.feature_names = self.X.columns.tolist()

if __name__ == "__main__":
    TrainingFlow()