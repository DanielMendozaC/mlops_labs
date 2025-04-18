"""
ML Training Flow using Metaflow

This script defines a Metaflow pipeline for training and registering
an ML model for wine quality prediction.
"""
from metaflow import FlowSpec, step, Parameter
import pandas as pd
import numpy as np

class WineQualityTrainingFlow(FlowSpec):
    """A flow for training a wine quality prediction model."""
    
    # Parameters that can be passed to the flow
    data_path = Parameter('data_path', 
                         default='../data/winequality-red.csv',
                         help='Path to the wine quality dataset')
    
    test_split = Parameter('test_split',
                          default=0.2,
                          help='Proportion of data to use for testing')
    
    random_state = Parameter('random_state',
                            default=42,
                            help='Random seed for reproducibility')
    
    # Models to try
    model_types = Parameter('model_types',
                           default='rf,gb',
                           help='Comma-separated list of models to try: rf (Random Forest), gb (Gradient Boosting)')
    
    # Target column name
    target_col = Parameter('target_col',
                          default='class',
                          help='Name of the target column in the dataset')
    
    @step
    def start(self):
        """Load and preprocess the wine quality dataset."""
        import pandas as pd
        from sklearn.model_selection import train_test_split
        import dataprocessing
        
        print(f"Loading data from {self.data_path}")
        
        # Use our data processing module to load the data
        data = dataprocessing.load_wine_data(self.data_path)
        
        # Preprocess with feature engineering and cleaning
        X, y = dataprocessing.preprocess_wine_data(
            data, 
            target_col=self.target_col,
            feature_engineering=True,
            clean_data=True
        )
        
        # Check if target was found
        if y is None:
            raise ValueError(f"Target column '{self.target_col}' not found in the dataset. Available columns: {data.columns.tolist()}")
        
        # Store column names for later use
        self.feature_names = list(X.columns)
        
        # Split into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_split, random_state=self.random_state
        )
        
        print(f"Train set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        
        # Parse model types
        self.models_to_train = self.model_types.split(',')
        print(f"Will train the following models: {self.models_to_train}")
        
        # Proceed to the train_models step with foreach
        self.next(self.train_models, foreach='models_to_train')
        
    @step
    def train_models(self):
        """Train a specific model type."""
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import mean_squared_error, r2_score
        
        model_name = self.input
        print(f"Training model: {model_name}")
        
        # Initialize the scaler to standardize features
        scaler = StandardScaler()
        
        # Create the appropriate model based on the model name
        if model_name == 'rf':
            model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
            self.model_fullname = 'Random Forest'
        elif model_name == 'gb':
            model = GradientBoostingRegressor(random_state=self.random_state)
            self.model_fullname = 'Gradient Boosting'
        else:
            raise ValueError(f"Unknown model type: {model_name}")
        
        # Create a pipeline with scaling and the model
        pipeline = Pipeline([
            ('scaler', scaler),
            ('model', model)
        ])
        
        # Train the model
        pipeline.fit(self.X_train, self.y_train)
        
        # Make predictions
        train_preds = pipeline.predict(self.X_train)
        test_preds = pipeline.predict(self.X_test)
        
        # Calculate metrics
        self.train_rmse = np.sqrt(mean_squared_error(self.y_train, train_preds))
        self.test_rmse = np.sqrt(mean_squared_error(self.y_test, test_preds))
        self.r2 = r2_score(self.y_test, test_preds)
        
        print(f"Model: {self.model_fullname}")
        print(f"Train RMSE: {self.train_rmse:.4f}")
        print(f"Test RMSE: {self.test_rmse:.4f}")
        print(f"R² Score: {self.r2:.4f}")
        
        # Store the trained pipeline
        self.pipeline = pipeline
        self.model_name = model_name
        
        # Move to the join step
        self.next(self.join)
    
    @step
    def join(self, inputs):
        """Join the parallel model training steps and select the best model."""
        # Get model results
        self.model_results = [(inp.model_name, inp.model_fullname, inp.test_rmse, inp.r2, inp.pipeline) 
                             for inp in inputs]
        
        # Sort models by test RMSE (lower is better)
        self.model_results = sorted(self.model_results, key=lambda x: x[2])
        
        # Select the best model (lowest RMSE)
        best_model = self.model_results[0]
        self.best_model_name = best_model[0]
        self.best_model_fullname = best_model[1]
        self.best_rmse = best_model[2]
        self.best_r2 = best_model[3]
        self.best_pipeline = best_model[4]
        
        print(f"Best model: {self.best_model_fullname} with Test RMSE: {self.best_rmse:.4f}")
        
        # Keep the feature names (from any input since they're all the same)
        self.feature_names = inputs[0].feature_names
        
        # Save X_test and y_test for model evaluation
        self.X_test = inputs[0].X_test
        self.y_test = inputs[0].y_test
        
        # Continue to the register_model step
        self.next(self.register_model)
    
    @step
    def register_model(self):
        """Register the best model with MLFlow."""
        import mlflow
        import mlflow.sklearn
        from datetime import datetime
        
        # Set up MLFlow tracking
        mlflow.set_tracking_uri('http://localhost:5000')  # Adjust if using a different URI
        mlflow.set_experiment("wine-quality-prediction")
        
        # Start a new MLFlow run
        with mlflow.start_run(run_name=f"metaflow-{self.best_model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}") as run:
            print(f"MLFlow run ID: {run.info.run_id}")
            
            # Log model parameters
            model_params = self.best_pipeline.named_steps['model'].get_params()
            for param_name, param_value in model_params.items():
                mlflow.log_param(param_name, param_value)
            
            # Log metrics
            mlflow.log_metric("test_rmse", self.best_rmse)
            mlflow.log_metric("r2_score", self.best_r2)
            
            # Log the feature names
            mlflow.log_param("features", self.feature_names)
            
            # Register the model
            mlflow.sklearn.log_model(
                self.best_pipeline,
                "model",
                registered_model_name="wine-quality-predictor"
            )
            
            print(f"Model registered with MLFlow as 'wine-quality-predictor'")
            
            # Store the run_id for future reference
            self.mlflow_run_id = run.info.run_id
        
        # Move to the end step
        self.next(self.end)
    
    @step
    def end(self):
        """Final step to display results and model information."""
        print("Training flow completed!")
        print(f"Best model: {self.best_model_fullname}")
        print(f"Model metrics:")
        print(f"- Test RMSE: {self.best_rmse:.4f}")
        print(f"- R² Score: {self.best_r2:.4f}")
        print(f"Model registered with MLFlow run ID: {self.mlflow_run_id}")
        
        # All model performance results
        print("\nAll models performance:")
        for model_name, fullname, rmse, r2, _ in self.model_results:
            print(f"- {fullname}: RMSE={rmse:.4f}, R²={r2:.4f}")

if __name__ == "__main__":
    WineQualityTrainingFlow()