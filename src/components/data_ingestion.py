import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method/component")
        try:
            # ✅ Make path independent of where you run the command from
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            csv_path = os.path.join(project_root, "notebook", "stud.csv")

            df = pd.read_csv(csv_path)
            logging.info(f"Read the dataset as dataframe from: {csv_path}")

            # ✅ Ensure artifacts directory exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train-test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of data is completed")

            # ✅ Visible terminal output (even if logger doesn't print to console)
            print("✅ Data ingestion completed")
            print("Train:", self.ingestion_config.train_data_path)
            print("Test :", self.ingestion_config.test_data_path)

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
