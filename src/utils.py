import os
import pickle
import sys
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import r2_score


def save_object(file_path, obj):
    try:
        logging.info("Dumping is started")

        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info('Dumping Completed')

    except Exception as e:
        logging.info("Error occured while dumping")
        raise CustomException(e, sys)


def evaluate_model(x_train, x_test, y_train, y_test, models):
    try:
        logging.info("Evaluation started")
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]

            logging.info(f"Training started for Model {i}")

            model.fit(x_train, y_train)

            y_test_pred = model.predict(x_test)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        logging.info('Error Occured while training model')
        raise CustomException(e, sys)


def load_object_pre(file_path):
    try:
        logging.info("Loading Started")
        logging.info(f"FilePAth at Utils {file_path}")
        x = r'C:\Users\Dell\Desktop\Python\Projects\Dimond_Price\artifacts\preprocessor.pkl'
        with open(x, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Error occurred While Loading in utils')
        raise CustomException(e, sys)


def load_object_model(file_path):
    try:
        logging.info("Loading Started")
        logging.info(f"FilePAth at Utils {file_path}")
        x = r'C:\Users\Dell\Desktop\Python\Projects\Dimond_Price\artifacts\model.pkl'
        with open(x, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Error occurred While Loading in utils')
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        logging.info("Loading Started")
        logging.info(f"FilePAth at Utils {file_path}")
        logging.info(f"PAth at utils{os.path.abspath(os.path.join(os.getcwd(), 'artifacts', 'model.pkl'))}")
        with open(file_path,"rb")as file_obj:
            # logging.info(f"Loading Completed {pickle.load(file_obj)}")
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Error occurred While Loading in utils')
        raise CustomException(e,sys)
