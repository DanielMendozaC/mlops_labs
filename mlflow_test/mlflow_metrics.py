import mlflow
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Set the tracking URI to the MLflow server running in Kubernetes
mlflow.set_tracking_uri('http://localhost:5001')

# Load wine dataset
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

# Create train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define hyperparameters to test
n_estimators_list = [50, 100, 200]
max_depth_list = [None, 10, 20]
min_samples_split_list = [2, 5, 10]

# Create experiment
mlflow.set_experiment("wine-classification-advanced")

# Track experiments
for n_estimators in n_estimators_list:
    for max_depth in max_depth_list:
        for min_samples_split in min_samples_split_list:
            with mlflow.start_run():
                # Log parameters
                mlflow.log_param("n_estimators", n_estimators)
                mlflow.log_param("max_depth", max_depth)
                mlflow.log_param("min_samples_split", min_samples_split)
                
                # Train model
                rf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42
                )
                rf.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred_train = rf.predict(X_train_scaled)
                y_pred_test = rf.predict(X_test_scaled)
                
                # Calculate metrics
                train_accuracy = accuracy_score(y_train, y_pred_train)
                test_accuracy = accuracy_score(y_test, y_pred_test)
                test_precision = precision_score(y_test, y_pred_test, average='weighted')
                test_recall = recall_score(y_test, y_pred_test, average='weighted')
                test_f1 = f1_score(y_test, y_pred_test, average='weighted')
                
                # Log metrics
                mlflow.log_metric("train_accuracy", train_accuracy)
                mlflow.log_metric("test_accuracy", test_accuracy)
                mlflow.log_metric("test_precision", test_precision)
                mlflow.log_metric("test_recall", test_recall)
                mlflow.log_metric("test_f1", test_f1)
                
                # Generate confusion matrix plot
                cm = confusion_matrix(y_test, y_pred_test)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Confusion Matrix')
                
                # Save figure to a file
                plt.savefig("confusion_matrix.png")
                plt.close()
                
                # Log figure
                mlflow.log_artifact("confusion_matrix.png")
                
                # Log feature importance
                feature_importance = pd.DataFrame(
                    rf.feature_importances_,
                    index=X.columns,
                    columns=['importance']
                ).sort_values('importance', ascending=False)
                
                # Create feature importance plot
                plt.figure(figsize=(10, 6))
                sns.barplot(x=feature_importance.importance, y=feature_importance.index)
                plt.title('Feature Importance')
                plt.tight_layout()
                plt.savefig("feature_importance.png")
                plt.close()
                
                # Log feature importance plot
                mlflow.log_artifact("feature_importance.png")
                
                # Log the model
                mlflow.sklearn.log_model(rf, "random_forest_model")
                
                # Log model signature with input and output schema
                signature = mlflow.models.infer_signature(
                    X_train_scaled, y_pred_train
                )
                mlflow.sklearn.log_model(
                    rf, 
                    "random_forest_model_with_signature",
                    signature=signature
                )
                
                print(f"Run completed with parameters: n_estimators={n_estimators}, "
                      f"max_depth={max_depth}, min_samples_split={min_samples_split}")
                print(f"Test accuracy: {test_accuracy:.4f}")

print("All experiments completed!")
print("Check MLflow UI to see the results")