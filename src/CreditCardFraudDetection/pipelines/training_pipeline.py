from src.CreditCardFraudDetection.components.data_ingestion import DataIngestion
# from src.CreditCardFraudDetection.components.data_transformation import DataTransformation
from src.CreditCardFraudDetection.components.model_training import ModelTraner

import pandas as pd
import os
import sys 
from src.CreditCardFraudDetection.logger import logging
from src.CreditCardFraudDetection.exception import CustomException

data_ingestio_obj = DataIngestion()

raw_data_path = data_ingestio_obj.initiate_data_ingestion()

model_training_obj = ModelTraner()
model_training_obj.initiate_model_traner(raw_data_path)
