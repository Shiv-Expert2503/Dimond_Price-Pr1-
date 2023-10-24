from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_model

import sys, os
from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'artifacts', 'model.pkl'))


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test array')
            x_train, x_test, y_train, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'L_R': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'Elastic': ElasticNet(),
                'DTRee': DecisionTreeRegressor(),
                'RForest': RandomForestRegressor()
            }

            model_report: dict = evaluate_model(x_train, y_train, x_test, y_test, models)
            logging.info(f"Model_Report : {model_report}")
            print("Model Report : ", model_report)
            print('\n')
            print('=' * 45)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f"Best Model : {best_model_name}, R2_Score : {best_model_score}")
            print('\n')
            print('=' * 45)
            logging.info(f"Best Model is {best_model_name}, R2_Score : {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )


        except Exception as e:
            logging.info('Error Occured while getting model report')
            raise CustomException(e, sys)
