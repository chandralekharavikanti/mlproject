import os
import sys

import numpy as np
import pandas as pd 
import dill 

from src.exception import CustomException

import os
import pickle

def save_object(file_path: str, obj) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)

    
