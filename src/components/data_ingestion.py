import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation


def get_project_root() -> str:
    # This file is at: <root>/src/components/data_ingestion.py
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@dataclass
class DataIngestionConfig:
    artifacts_dir: str = os.path.join(get_project_root(), "artifacts")
    train_data_path: str = os.path.join(get_project_root(), "artifacts", "train.csv")
    test_data_path: str = os.path.join(get_project_root(), "artifacts", "test.csv")
    raw_data_path: str = os.path.join(get_project_root(), "artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method/component")
        try:
            project_root = get_project_root()
            csv_path = os.path.join(project_root, "notebook", "stud.csv")

            df = pd.read_csv(csv_path)
            logging.info(f"Read the dataset as dataframe from: {csv_path}")

            # Normalize columns once, early (spaces + slashes)
            df.columns = (
                df.columns.str.strip()
                .str.replace(" ", "_")
                .str.replace("/", "_")
            )

            #  Ensure artifacts directory exists (absolute)
            os.makedirs(self.ingestion_config.artifacts_dir, exist_ok=True)

            # Save raw
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train-test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of data is completed")

            print("Data ingestion completed")
            print("Train:", self.ingestion_config.train_data_path)
            print("Test :", self.ingestion_config.test_data_path)

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    print("calling datatransformation")

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)
