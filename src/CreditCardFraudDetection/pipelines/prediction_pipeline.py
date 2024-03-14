import sys
import os
import pandas as pd

from src.CreditCardFraudDetection.exception import CustomException
from src.CreditCardFraudDetection.logger import logging
from src.CreditCardFraudDetection.utils.utils import load_object

class Predict_Pipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")

            logging.info("reading the model object from artifacts")

            # preprocessor_obj = load_object(preprocessor_path)
            model_obj = load_object(model_path)

            # logging.info("transoforming the new data")

            # scaled_data = preprocessor_obj.transform(features)

            logging.info("Predicting the price of the Flight")

            prediction = model_obj.predict(features)

            return prediction

        except Exception as e:
            logging.info("Exceptio while Predicing")
            raise CustomException(e, sys)


class CustomData:

    def __init__(self, Time,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,
                 V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,Amount):
        self.Time = Time
        self.V1 = V1
        self.V2 = V2
        self.V3 = V3
        self.V4 = V4
        self.V5 = V5
        self.V6 = V6
        self.V7 = V7
        self.V8 = V8
        self.V9 = V9
        self.V10 = V10
        self.V11 = V11
        self.V12 = V12
        self.V13 = V13
        self.V14 = V14
        self.V15 = V15
        self.V16 = V16
        self.V17 = V17
        self.V18 = V18
        self.V19 = V19
        self.V20 = V20
        self.V21 = V21
        self.V22 = V22
        self.V23 = V23
        self.V24 = V24
        self.V25 = V25
        self.V26 = V26
        self.V27 = V27
        self.V28 = V28
        self.Amount = Amount
        

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Time':[self.Time],
                'V1':[self.V1],
                'V2':[self.V2],
                'V3':[self.V3],
                'V4':[self.V4],
                'V5':[self.V5],
                'V6':[self.V6],
                'V7':[self.V7],
                'V8':[self.V8],
                'V9':[self.V9],
                'V10':[self.V10],
                'V11':[self.V11],
                'V12':[self.V12],
                'V13':[self.V13],
                'V14':[self.V14],
                'V15':[self.V15],
                'V16':[self.V16],
                'V17':[self.V17],
                'V18':[self.V18],
                'V19':[self.V19],
                'V20':[self.V20],
                'V21':[self.V21],
                'V22':[self.V22],
                'V23':[self.V23],
                'V24':[self.V24],
                'V25':[self.V25],
                'V26':[self.V26],
                'V27':[self.V27],
                'V28':[self.V28],
                'Amount':[self.Amount]
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info("DataFrame Created from the new Data")

            return df
        
        except Exception as e:
            logging.info("Exception in get_data_as_dataframe()")
            raise CustomException(e, sys)
        
if __name__=="__main__":
    sent = "0.0,-1.3598071336738,-0.0727811733098497,2.53634673796914,1.37815522427443,-0.338320769942518,0.462387777762292,0.239598554061257,0.0986979012610507,0.363786969611213,0.0907941719789316,-0.551599533260813,-0.617800855762348,-0.991389847235408,-0.311169353699879,1.46817697209427,-0.470400525259478,0.207971241929242,0.0257905801985591,0.403992960255733,0.251412098239705,-0.018306777944153,0.277837575558899,-0.110473910188767,0.0669280749146731,0.128539358273528,-0.189114843888824,0.133558376740387,-0.0210530534538215,149.62"
    lst = sent.split(',')
    cd = CustomData(lst[0],
                    lst[1],
                    lst[2],
                    lst[3],
                    lst[4],
                    lst[5],
                    lst[6],
                    lst[7],
                    lst[8],
                    lst[9],
                    lst[10],
                    lst[11],
                    lst[12],
                    lst[12],
                    lst[14],
                    lst[15],
                    lst[16],
                    lst[17],
                    lst[18],
                    lst[19],
                    lst[20],
                    lst[21],
                    lst[22],
                    lst[23],
                    lst[24],
                    lst[25],
                    lst[26],
                    lst[27],
                    lst[28],
                    lst[29])
    df = cd.get_data_as_dataframe()

    pred = Predict_Pipeline()
    predicted_value = pred.predict(df)
    print("=================",predicted_value)