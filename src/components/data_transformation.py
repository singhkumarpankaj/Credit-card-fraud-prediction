from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.preprocessing import StandardScaler # HAndling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
import sys,os
from dataclasses import dataclass

from src.utils import save_obj

@dataclass
class DatatranformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DatatranformationConfig()

    def get_data_transformation_obj(self):
        try:
            logging.info("Data Transformation Pipeline Initiated")

            numerical_cols = ['LIMIT_BAL', 'SEX','EDUCATION','MARRIAGE','AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6','BILL_AMT1',
            'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5','PAY_AMT6']

            categorical_cols = []
            
            ## Numerical Pipeline
            num_pipeline = Pipeline(
                steps = [
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('scaler',StandardScaler())
                ]

            )

            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])

            logging.info("Data Transformation is completed")

            return preprocessor

        except Exception as e:
            logging.info("Exception occured in Data Transformation")
            raise CustomException(e, sys)


    def initiate_data_transformation(self,train_data_path,test_data_path):
        
        try:
            train_df = pd.read_csv(train_data_path)
            test_df=pd.read_csv(test_data_path)

            logging.info('Reading the train and test data completed')
            logging.info(f'Training Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')

            logging.info('Obtaining Preprocessing Object')

            preprocessing_obj=self.get_data_transformation_obj()

            target_column= ["default payment next month"]

            #Dividing into dependent and independent features

            #Training Data
            input_feature_train_df = train_df.drop(columns= target_column,axis=1)
            target_feature_train_df = train_df[target_column]
            #Test Data
            input_feature_test_df = test_df.drop(columns= target_column,axis=1)
            target_feature_test_df = test_df[target_column]

            #Data Transformation
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

            logging.info("Applying preprocessing object on training and testing datasets.")


        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e, sys)