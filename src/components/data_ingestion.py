import os
import sys
import pandas as pd
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_train_path: str = os.path.join('artifacts', "raw_train.csv")
    raw_test_path: str = os.path.join('artifacts', "raw_test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Read from mlproject/data/ (correct path)
            df_train = pd.read_csv('notebook/data/train.csv')
            df_test = pd.read_csv('notebook/data/test.csv')
            
            logging.info(f"Read train dataset: {df_train.shape}")
            logging.info(f"Read test dataset: {df_test.shape}")

            # Create artifacts directory
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            logging.info("Created artifacts/ directory")

            # Save raw data (for reproducibility)
            df_train.to_csv(self.ingestion_config.raw_train_path, index=False, header=True)
            df_test.to_csv(self.ingestion_config.raw_test_path, index=False, header=True)

            # Save processed data (what transformation expects)
            df_train.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            df_test.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion complete")
            logging.info(f"Train saved: {self.ingestion_config.train_data_path}")
            logging.info(f"Test saved: {self.ingestion_config.test_data_path}")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))
