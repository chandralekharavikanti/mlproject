import os
import sys

import numpy as np
import pandas as pd 
import dill 

from sklearn.metrics import r2_score
from src.exception import CustomException

import os
import pickle

def save_object(file_path: str, obj) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)

from sklearn.metrics import r2_score

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for name, model in models.items():
            
            model.fit(X_train, y_train)

            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            report[name] = test_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
