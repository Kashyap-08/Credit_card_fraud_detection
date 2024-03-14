from src.CreditCardFraudDetection.logger import logging
from src.CreditCardFraudDetection.exception import CustomException


import os
import pickle
import sys
from sklearn.metrics import r2_score, classification_report, accuracy_score


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        logging.info(f"Make Director named: {dir_path}")
        os.makedirs(dir_path, exist_ok=True)

        logging.info('Directory Created')

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

        logging.info(f"Dumped Object in filepath: {file_path}")

    except Exception as e:
        logging.info("Exceptio in save_object() method")
        raise CustomException(e, sys)
    

def evaluate_model(X_train, y_train, X_test, y_test, models):
    try: 
        report = {}
        for i, (model_name,model) in enumerate(models.items()):
            if model_name == "Local Outlier Factor":
                y_train_pred = model.fit_predict(X_train)
                y_test_pred = model.fit_predict(X_test)
            
            elif model_name == "Random Forest Regressor":
                model.fit(X_train, y_train)
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)  
                
            else:
                model.fit(X_train)
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)


            #y_train_pred = model.predict(X_train)
            y_train_pred[y_train_pred == 1] = 0
            y_train_pred[y_train_pred == -1] = 1
            train_accuracy = accuracy_score(y_train, y_train_pred)

            # y_test_pred = model.predict(X_test)
            y_test_pred[y_test_pred == 1] = 0
            y_test_pred[y_test_pred == -1] = 1
            test_accuracy = accuracy_score(y_test, y_test_pred)

            train_error_points =  (y_train_pred != y_train).sum()
            test_error_points =  (y_test_pred != y_test).sum()

            print("Accuracy Score for Train:", train_accuracy)
            logging.info(f"Accuracy Score for Train: {train_accuracy}")

            print("Accuracy Score for Test:", test_accuracy)
            logging.info(f"Accuracy Score for Test: {test_accuracy}")

            print('Error Points for ', model_name)
            logging.info(f"Error Points for :{model_name}")

            print("Error for Train:", train_error_points)
            logging.info(f"Error for Train::{train_error_points}")

            print("Error for Test:", test_error_points)
            logging.info(f"Error  for Test :{test_error_points}")

            print("Classification Report")
            print(classification_report(y_test, y_test_pred))
            print("\n===================================================\n")

            logging.info("Classification Report")
            logging.info(classification_report(y_test, y_test_pred))
            logging.info("\n===================================================\n")

            report[model_name] = test_accuracy

        return report
    
    except Exception as e:
        logging.info("Exception in evaluate_model()")
        raise CustomException(e, sys)
    

def load_object(path):
    try:
        with open(path, 'rb') as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        logging.info('Exception in load_object()')
        raise CustomException(e, sys)