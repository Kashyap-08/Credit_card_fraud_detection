import pandas as pd
import numpy as np
import sys
from src.CreditCardFraudDetection.logger import logging
from src.CreditCardFraudDetection.exception import CustomException

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path
import os

class DataIngestionConfig:
    raw_data_path = os.path.join("artifacts", "raw.csv")
    train_data_path = os.path.join("artifacts", "train.csv")
    test_data_path = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Start Data Ingestion")

        try:
            data = pd.read_csv(Path(os.path.join("notebooks/data", "creditcard.csv")))
            logging.info("read data in the DataFrame fromat")

            # Store Raw data in Artifact Folder
            raw_data_path = self.ingestion_config.raw_data_path
            os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)   # Make Director if not exist
            data.to_csv(raw_data_path, index=False)
            logging.info(f"Stored Raw data in {raw_data_path}")

            return (
                raw_data_path
            )


        except Exception as e:
            logging.info("Exception During the data ingestion process")
            raise CustomException(e, sys)