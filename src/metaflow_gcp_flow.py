from metaflow import FlowSpec, step, conda_base, kubernetes, timeout, retry, catch, Parameter
import os

@conda_base(libraries={'numpy':'1.23.5', 'scikit-learn':'1.2.2', 'pandas':'1.5.3', 'mlflow':'2.3.0'}, python='3.9.16')
class MLTrainingFlowGCP(FlowSpec):
    """
    A flow to train machine learning models in GCP using Kubernetes.
    This flow demonstrates how to set up a scalable training pipeline that integrates with MLFlow.
    """
    
    # Define parameters for your flow
    model_name = Parameter('model_name', default='lasso_model')
    experiment_name = Parameter('experiment_name', default='wine_classification')
    
    @step
    def start(self):
        """
        Load and prepare the dataset.
        """
        # Import libraries here to ensure they're in the conda environment
        from sklearn import datasets
        from sklearn.model_selection import train_test_split
        import numpy as np
        
        print("Starting the ML training pipeline in GCP...")
        
        # Load sample dataset (replace with your own data loading logic)
        X, y = datasets.load_wine(return_X_y=True)
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Data loaded successfully: {self.train_data.shape[0]} training samples")
        
        # Define hyperparameter ranges for model training
        self.alphas = np.logspace(-3, 0, 5)  # 5 different alpha values
        
        # Branch out for each alpha value
        self.next(self.train_model, foreach='alphas')
    
    @kubernetes
    @retry(times=3)
    @timeout(minutes=10)
    @catch(var='train_error')
    @step
    def train_model(self):
        """
        Train a model with the current alpha parameter.
        This step runs in Kubernetes for scalability.
        """
        import mlflow
        from sklearn.linear_model import Lasso
        import numpy as np
        
        # Set MLflow tracking URI to your MLflow server
        mlflow.set_tracking_uri("http://your-mlflow-server-ip:5000")  # Replace with your MLflow server
        
        # Check if there was an error
        if self.train_error:
            print(f"Error in training: {self.train_error}")
            return
        
        # Current alpha value from the foreach loop
        self.alpha = self.input
        print(f"Training model with alpha={self.alpha}")
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"{self.model_name}_alpha_{self.alpha}") as run:
            # Set experiment
            mlflow.set_experiment(self.experiment_name)
            
            # Train model
            self.model = Lasso(alpha=self.alpha, max_iter=10000)
            self.model.fit(self.train_data, self.train_labels)
            
            # Evaluate model
            train_score = self.model.score(self.train_data, self.train_labels)
            test_score = self.model.score(self.test_data, self.test_labels)
            
            # Log parameters and metrics
            mlflow.log_param("alpha", self.alpha)
            mlflow.log_metric("train_score", train_score)
            mlflow.log_metric("test_score", test_score)
            
            # Save model coefficients for inspection
            self.coef = self.model.coef_
            self.intercept = self.model.intercept_
            
            # Save run_id for later reference
            self.run_id = run.info.run_id
            
            print(f"Model trained successfully. Test score: {test_score}")
            
            # Log the model to MLflow
            mlflow.sklearn.log_model(self.model, "model")
        
        self.next(self.join_results)
    
    @step
    def join_results(self, inputs):
        """
        Join all parallel runs and select the best model.
        """
        # Collect results from all models
        self.models = []
        self.scores = []
        self.run_ids = []
        
        for inp in inputs:
            self.models.append(inp.model)
            self.scores.append(inp.model.score(inp.test_data, inp.test_labels))
            self.run_ids.append(inp.run_id)
        
        # Find the best model
        best_idx = self.scores.index(max(self.scores))
        self.best_model = self.models[best_idx]
        self.best_score = self.scores[best_idx]
        self.best_run_id = self.run_ids[best_idx]
        self.best_alpha = inputs[best_idx].alpha
        
        print(f"Best model found: alpha={self.best_alpha}, score={self.best_score}")
        
        self.next(self.register_model)
    
    @step
    def register_model(self):
        """
        Register the best model in MLflow model registry.
        """
        import mlflow
        
        # Set MLflow tracking URI to your MLflow server
        mlflow.set_tracking_uri("http://your-mlflow-server-ip:5001")  # Replace with your MLflow server
        
        # Register the model
        model_uri = f"runs:/{self.best_run_id}/model"
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=f"{self.model_name}_production"
        )
        
        print(f"Model registered successfully: {registered_model.name}, version: {registered_model.version}")
        
        self.next(self.end)
    
    @step
    def end(self):
        """
        Final step to display results.
        """
        print("ML Training Pipeline completed successfully!")
        print(f"Best model (alpha={self.best_alpha}) registered with score: {self.best_score}")
        print(f"Model available in MLflow with run_id: {self.best_run_id}")


if __name__ == "__main__":
    MLTrainingFlowGCP()