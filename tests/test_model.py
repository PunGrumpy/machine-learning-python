import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model.model import train_model
from src.utils.load_and_save import load_model, save_model
import numpy as np


def test_train_model():
    model, X_test, y_test = train_model()
    assert model is not None, "Model training failed"


def test_model_prediction():
    model, X_test, y_test = train_model()
    predictions = model.predict(X_test)
    assert predictions is not None, "Model prediction failed"
    assert len(predictions) == len(
        y_test
    ), "Number of predictions doesn't match with number of test samples"


def test_save_and_load_model():
    model, _, _ = train_model()
    save_model(model, "models/test_model.pkl")
    loaded_model = load_model("models/test_model.pkl")
    assert loaded_model is not None, "Model loading failed"
