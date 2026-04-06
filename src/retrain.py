import mlflow
import mlflow.sklearn
import joblib
import os
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlflow.tracking import MlflowClient

THRESHOLD = 0.90

def get_production_accuracy():
    client = MlflowClient()
    try:
        runs = client.search_runs(
            experiment_ids=["1"],
            order_by=["metrics.accuracy DESC"],
            max_results=1
        )
        if not runs:
            return 0.0
        return float(runs[0].data.metrics.get("accuracy", 0))
    except:
        return 0.0

def retrain():
    mlflow.set_experiment("auto_retrain")

    X, y = load_iris(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=None
    )

    with mlflow.start_run(run_name="auto_retrain_run"):
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, model.predict(X_te))
        prod_acc = get_production_accuracy()

        mlflow.log_metric("new_accuracy", acc)
        mlflow.log_metric("prod_accuracy", prod_acc)
        mlflow.log_param("trigger", "scheduled")

        if acc >= prod_acc and acc >= THRESHOLD:
            mlflow.sklearn.log_model(model, "model")
            os.makedirs("models", exist_ok=True)
            joblib.dump(model, "models/best_model.pkl")
            print(f"New model saved! accuracy={acc:.3f}")
        else:
            print(f"Kept existing model. new={acc:.3f}, prod={prod_acc:.3f}")

if __name__ == "__main__":
    retrain()