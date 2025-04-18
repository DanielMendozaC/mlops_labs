"""
ML Scoring Flow using Metaflow

This script defines a Metaflow pipeline for making predictions using 
a trained model that was registered with MLFlow.
"""
from metaflow import FlowSpec, step, Parameter
import pandas as pd
import numpy as np

class WineQualityScoringFlow(FlowSpec):
    """A flow for scoring wine quality using a trained model."""
    
    # Parameters
    data_path = Parameter('data_path', 
                         default='../data/wine_samples.csv',
                         help='Path to data for scoring')
    
    model_name = Parameter('model_name',
                          default='wine-quality-predictor',
                          help='Name of the registered model in MLFlow')
    
    model_stage = Parameter('model_stage',
                           default='None',
                           help='Stage of the model to use (Production, Staging, None)')
    
    output_path = Parameter('output_path',
                           default='../data/predictions.csv',
                           help='Path to save predictions')
    
    # Target column name
    target_col = Parameter('target_col',
                          default='class',
                          help='Name of the target column in the dataset')
    
    @step
    def start(self):
        """Load data for scoring."""
        import pandas as pd
        import dataprocessing
        
        print(f"Loading data from {self.data_path}")
        
        # Use our data processing module to load the data
        self.data = dataprocessing.load_wine_data(self.data_path)
        
        # Check if target column exists - it should be removed for inference
        self.has_target = self.target_col in self.data.columns
        
        # Process the data using our utility functions
        self.X, y_processed = dataprocessing.preprocess_wine_data(
            self.data, 
            target_col=self.target_col if self.has_target else None,
            feature_engineering=True,
            clean_data=True
        )
        
        if self.has_target:
            self.y_true = y_processed
            print(f"Found target column '{self.target_col}' - will use for evaluation")
        else:
            print(f"No target column '{self.target_col}' found - will only generate predictions")
            
        # Move to the next step to load the model
        self.next(self.load_model)
        
    @step
    def load_model(self):
        """Load the trained model from MLFlow."""
        import mlflow
        import mlflow.sklearn
        
        # Set up MLFlow tracking
        mlflow.set_tracking_uri('http://localhost:5000')  # Adjust if using a different URI
        
        print(f"Loading model '{self.model_name}' from MLFlow registry")
        
        # Load the model - if stage is None, get the latest version
        if self.model_stage.lower() == 'none':
            self.model_uri = f"models:/{self.model_name}/latest"
        else:
            self.model_uri = f"models:/{self.model_name}/{self.model_stage}"
            
        try:
            self.model = mlflow.sklearn.load_model(self.model_uri)
            print(f"Model loaded successfully from {self.model_uri}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
            
        # Continue to prediction step
        self.next(self.predict)
        
    @step
    def predict(self):
        """Generate predictions using the loaded model."""
        from sklearn.metrics import mean_squared_error, r2_score
        
        print("Generating predictions...")
        
        # Make predictions
        self.predictions = self.model.predict(self.X)
        
        # Create a results DataFrame
        results_df = self.X.copy()
        results_df['predicted_class'] = self.predictions
        
        # If we have true values, add them and calculate metrics
        if self.has_target:
            results_df['actual_class'] = self.y_true
            
            # Calculate metrics
            self.rmse = np.sqrt(mean_squared_error(self.y_true, self.predictions))
            self.r2 = r2_score(self.y_true, self.predictions)
            
            print(f"Prediction metrics:")
            print(f"- RMSE: {self.rmse:.4f}")
            print(f"- R² Score: {self.r2:.4f}")
        
        # Save the results
        self.results_df = results_df
        
        # Continue to save step
        self.next(self.save_results)
        
    @step
    def save_results(self):
        """Save the predictions to a CSV file."""
        print(f"Saving predictions to {self.output_path}")
        
        # Save predictions
        try:
            self.results_df.to_csv(self.output_path, index=False)
            print(f"Predictions saved successfully")
        except Exception as e:
            print(f"Error saving predictions: {e}")
            raise
            
        # Continue to end
        self.next(self.end)
        
    @step
    def end(self):
        """Final step with summary of the scoring process."""
        print("Scoring flow completed!")
        print(f"Processed {len(self.predictions)} samples")
        
        if self.has_target:
            print(f"Prediction performance:")
            print(f"- RMSE: {self.rmse:.4f}")
            print(f"- R² Score: {self.r2:.4f}")
            
        print(f"Results saved to: {self.output_path}")

if __name__ == "__main__":
    WineQualityScoringFlow()