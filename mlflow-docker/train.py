# train.py
import mlflow, mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

mlflow.set_experiment("Default")  
with mlflow.start_run():
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = ((preds - y_test)**2).mean()**0.5

    mlflow.log_param("n_estimators", model.n_estimators)
    mlflow.log_metric("rmse", rmse)
    mlflow.sklearn.log_model(model, "model")
    print("Run logged with RMSE:", rmse)
