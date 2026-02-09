import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


def get_project_root() -> str:
    # This file is at: <root>/src/components/data_transformation.py
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join(
        get_project_root(), "artifacts", "preprocessor.pkl"
    )


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                    # OneHotEncoder outputs sparse -> with_mean=False
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ],
                remainder="drop",
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            print("Entered initiate_data_transformation()")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Safety: normalize again (in case files came from elsewhere)
            train_df.columns = (
                train_df.columns.str.strip()
                .str.replace(" ", "_")
                .str.replace("/", "_")
            )
            test_df.columns = (
                test_df.columns.str.strip()
                .str.replace(" ", "_")
                .str.replace("/", "_")
            )

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"

            # Helpful explicit check (gives clear error)
            if target_column_name not in train_df.columns:
                raise ValueError(
                    f"Target column '{target_column_name}' not found. Columns: {train_df.columns.tolist()}"
                )

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            print("Saved to:", os.path.abspath(self.data_transformation_config.preprocessor_obj_file_path))
            print("Exists?", os.path.exists(self.data_transformation_config.preprocessor_obj_file_path))

            logging.info(
                f"Preprocessor saved at: {os.path.abspath(self.data_transformation_config.preprocessor_obj_file_path)}"
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
