import pickle
import numpy as np
import os


def save_model(model, filepath):
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    with open(filepath, "wb") as f:
        pickle.dump(model, f)


def load_model(filepath):
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    return model


def save_numpy_array(array, filepath):
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    with open(filepath, "wb") as f:
        np.save(f, array)


def load_numpy_array(filepath):
    with open(filepath, "rb") as f:
        array = np.load(f)
    return array
