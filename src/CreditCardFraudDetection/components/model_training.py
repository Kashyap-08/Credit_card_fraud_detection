import pandas as pd
import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor

from src.CreditCardFraudDetection.logger import logging
from src.CreditCardFraudDetection.exception import CustomException
from src.CreditCardFraudDetection.utils.utils import save_object, evaluate_model

from sklearn.ensemble import IsolationForest,  RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


@dataclass
class ModelTrainingConfig:
    training_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTraner:
    def __init__(self):
        self.model_training_config = ModelTrainingConfig()

    # def initiate_model_traner(self, train_array, test_array):
    def initiate_model_traner(self, data_path):
        try:
            logging.info('Spliting Dependent and Independent variables from train and test data')

            data = pd.read_csv(data_path)
            data = data.iloc[:10000,:]
            Fraud = data[data.Class == 1]
            Valid = data[data.Class == 0]
            outlier_fraction = len(Fraud)/float(len(Valid))

            X = data.drop("Class", axis=1)
            y = data['Class']

            # splitting the data into train and test 
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=33)

            models ={
                "Isolation Forest":IsolationForest(n_estimators=100, max_samples=len(X), 
                                                contamination=outlier_fraction,random_state=42, verbose=0),
                "Local Outlier Factor":LocalOutlierFactor(n_neighbors=20, algorithm='auto', 
                                                        leaf_size=30, metric='minkowski',
                                                        p=2, metric_params=None, contamination=outlier_fraction),
                "Support Vector Machine":OneClassSVM(kernel='rbf', degree=3, gamma=0.1,nu=0.05, 
                                                    max_iter=-1),
                "Random Forest Regressor": RandomForestClassifier()
            }


            model_report:dict = evaluate_model(X_train, y_train, X_test, y_test, models)
            logging.info(f"Model Report: {model_report}")

            # to get the best model score from dictionary
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]
            print("\n =========================================================== \n")
            print(f"Best model found, Model Name: {best_model_name}, R2_Score: {best_model_score}")
            print("\n =========================================================== \n")
            logging.info(f"Best model found, Model Name: {best_model_name}, R2_Score: {best_model_score}")

            save_object(
                file_path = self.model_training_config.training_model_file_path,
                obj = best_model
            )

            logging.info('Trained Object Saved Successfully.')
        except Exception as e:
            logging.info('Exception in initiate_model_trainer() method')
            raise CustomException(e, sys)