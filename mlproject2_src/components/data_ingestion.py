import os
import sys
import pandas as pd
from dataclasses import dataclass

from mlproject2_src.exception import CustomException
from mlproject2_src.logger import logging
from mlproject2_src.components.data_transformation import DataTransformation
from mlproject2_src.components.model_trainer import ModelTrainer



@dataclass
class DataIngestionConfig:
    artifacts_dir: str = os.path.join("artifacts")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_train_path: str = os.path.join("artifacts", "raw_train.csv")
    raw_test_path: str = os.path.join("artifacts", "raw_test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component")
        try:
            train_candidates = [os.path.join("data", "train.csv"), os.path.join("notebook", "data", "train.csv")]
            test_candidates = [os.path.join("data", "test.csv"), os.path.join("notebook", "data", "test.csv")]

            train_path = next((p for p in train_candidates if os.path.exists(p)), None)
            test_path = next((p for p in test_candidates if os.path.exists(p)), None)

            if train_path is None or test_path is None:
                raise FileNotFoundError(
                    "Could not find train.csv/test.csv in data/ or notebook/data/. "
                    "Place files in data/ (recommended) or notebook/data/."
                )

            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)

            os.makedirs(self.ingestion_config.artifacts_dir, exist_ok=True)

            df_train.to_csv(self.ingestion_config.raw_train_path, index=False)
            df_test.to_csv(self.ingestion_config.raw_test_path, index=False)

            df_train.to_csv(self.ingestion_config.train_data_path, index=False)
            df_test.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Data ingestion completed")
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    X_train, y_train, X_test, _ = data_transformation.initiate_data_transformation(train_data, test_data)
    print("Data transformation completed.")


    model_trainer = ModelTrainer()
    result = model_trainer.initiate_model_trainer(X_train, y_train)

    print(result)
