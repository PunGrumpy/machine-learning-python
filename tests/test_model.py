# tests/test_model.py
import unittest
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import train_model, save_model
from src.validate_model import validate_model


class TestModel(unittest.TestCase):
    def setUp(self):
        self.X, self.y = load_diabetes(return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        self.model = LinearRegression()

    def test_train_model(self):
        model, X_test, y_test = train_model()
        self.assertIsNotNone(model)
        self.assertIsNotNone(X_test)
        self.assertIsNotNone(y_test)

    def test_save_model(self):
        if not os.path.exists("models/"):
            os.makedirs("models/")
        save_model(self.model, "models/test_model.pkl")
        self.assertTrue(os.path.exists("models/test_model.pkl"))
        os.remove("models/test_model.pkl")

    def test_validate_model(self):
        self.model.fit(self.X_train, self.y_train)
        mse = validate_model(self.model, self.X_test, self.y_test)
        self.assertLess(mse, 3000.0)  # Changed the threshold to 3000


if __name__ == "__main__":
    unittest.main()
