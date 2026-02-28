import os
import sys
import pickle
from sklearn.metrics import r2_score


def save_object(file_path: str, obj) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
    except Exception as e:
        from src.exception import CustomException
        raise CustomException(e, sys)


def load_object(file_path: str):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        from src.exception import CustomException
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models: dict):
    try:
        report = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            report[name] = r2_score(y_test, y_test_pred)
        return report
    except Exception as e:
        from src.exception import CustomException
        raise CustomException(e, sys)