import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model_path = os.path.join('artifacts','model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)
            return pred

        except CustomException as e :
            logging.info("error in the prediction pipeline")
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,LIMIT_BAL:int,SEX:int,EDUCATION:int,MARRIAGE:int,AGE:int,PAY_0:int,PAY_2:int,PAY_3:int,PAY_4:int,PAY_5:int,PAY_6:int,BILL_AMT1:int,BILL_AMT2:int,BILL_AMT3:int,BILL_AMT4:int,BILL_AMT5:int,BILL_AMT6:int,PAY_AMT1:int,PAY_AMT2:int,PAY_AMT3:int,PAY_AMT4:int,PAY_AMT5:int,PAY_AMT6:int):
        self.LIMIT_BAL = LIMIT_BAL
        self.SEX = SEX
        self.EDUCATION = EDUCATION
        self.MARRIAGE = MARRIAGE
        self.AGE = AGE 
        self.PAY_0 = PAY_0
        self.PAY_2 = PAY_2
        self.PAY_3 = PAY_3
        self.PAY_4 = PAY_4
        self.PAY_5 = PAY_5
        self.PAY_6 = PAY_6
        self.BILL_AMT1 = BILL_AMT1
        self.BILL_AMT2 = BILL_AMT2
        self.BILL_AMT3 = BILL_AMT3
        self.BILL_AMT4 = BILL_AMT4
        self.BILL_AMT5 = BILL_AMT5
        self.BILL_AMT6 = BILL_AMT6
        self.PAY_AMT1 = PAY_AMT1
        self.PAY_AMT2 = PAY_AMT2
        self.PAY_AMT3 = PAY_AMT3
        self.PAY_AMT4 = PAY_AMT4
        self.PAY_AMT5 = PAY_AMT5
        self.PAY_AMT6 = PAY_AMT6

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'LIMIT_BAL' : [self.LIMIT_BAL],
                'SEX' : [self.SEX],
                'EDUCATION' : [self.EDUCATION],
                'MARRIAGE' : [self.MARRIAGE],
                'AGE' : [self.AGE],
                'PAY_0' : [self.PAY_0],
                'PAY_2' : [self.PAY_2],
                'PAY_3' : [self.PAY_3],
                'PAY_4' : [self.PAY_4],
                'PAY_5' : [self.PAY_5],
                'PAY_6' : [self.PAY_6],
                'BILL_AMT1' : [self.BILL_AMT1],
                'BILL_AMT2' : [self.BILL_AMT2],
                'BILL_AMT3' : [self.BILL_AMT3],
                'BILL_AMT4' : [self.BILL_AMT4],
                'BILL_AMT5' : [self.BILL_AMT5],
                'BILL_AMT6' : [self.BILL_AMT6],
                'PAY_AMT1' : [self.PAY_AMT1],
                'PAY_AMT2' : [self.PAY_AMT2],
                'PAY_AMT3' : [self.PAY_AMT3],
                'PAY_AMT4' : [self.PAY_AMT4],
                'PAY_AMT5' : [self.PAY_AMT5],
                'PAY_AMT6' : [self.PAY_AMT6]
                }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('DataFrame Gathered')
            return df
        except Exception as e:
            logging.info("Exception Occured in Getting data as dataframe from the prediction pipeline")
            raise CustomException(e, sys)