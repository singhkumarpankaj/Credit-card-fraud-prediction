import os,sys
import pickle
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

def save_obj(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        logging.info("Error in saving the pickle file")
        raise CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}  # Use a dictionary to store the results
        for model_name, model in models.items():
            # Train Model
            model.fit(X_train, y_train)

            # Predicting the testing data
            y_test_pred = model.predict(X_test)

            # Getting Accuracy for test data
            test_model_score = accuracy_score(y_test, y_test_pred)

            # Store the model name and its test score in the report dictionary
            report[model_name] = test_model_score

        return report

    except Exception as e:
        logging.info("Exception occurred training model")
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as obj:
            return pickle.load(obj)

    except Exception as e:
        logging.info('Exception occured in load_object function utils')
        raise CustomException(e,sys)