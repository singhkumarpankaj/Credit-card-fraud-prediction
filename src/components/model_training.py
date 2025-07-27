import numpy as np
import pandas as pd
import sys,os

## Model libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj,evaluate_model
from dataclasses import dataclass

@dataclass
class ModelTrainerconfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerconfig()

    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info("Splitting Dependent and independent features from train and test data")
            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            logging.info("Splitting of independent and dependent train and test data done")

            models = {
                "Logistic Regression": LogisticRegression(),
                "Support Vector Machine": SVC(),
                "Naive Bayes": GaussianNB(),
                "Random Forest": RandomForestClassifier(max_depth=10),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Decision Tree": DecisionTreeClassifier(max_depth=10),
                "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
                }

            model_report:dict= evaluate_model(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print("\n")
            print('='*35)
            logging.info(f'Model Report : {model_report}')

            #To get the best model score from dictionary
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f"Best Model Found , Model Name : {best_model_name} , Accuracy : {best_model_score}")
            print("\n")
            print("="*35)
            logging.info(f"Best Model Found , Model Name : {best_model_name} , Accuracy : {best_model_score}")

            save_obj(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj=best_model
                
            )

        except Exception as e:
            logging.info("Error in Model Training")
            raise CustomException(e, sys)