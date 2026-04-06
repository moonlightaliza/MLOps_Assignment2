import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

def test_model_loads():
    model = joblib.load("models/best_model.pkl")
    assert model is not None

def test_model_predicts():
    model = joblib.load("models/best_model.pkl")
    sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    pred = model.predict(sample)
    assert pred[0] in [0, 1, 2]

def test_model_accuracy():
    model = joblib.load("models/best_model.pkl")
    X, y = load_iris(return_X_y=True)
    acc = accuracy_score(y, model.predict(X))
    assert acc > 0.90