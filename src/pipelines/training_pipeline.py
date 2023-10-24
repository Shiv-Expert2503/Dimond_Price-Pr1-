import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import Data_Transformation
from src.components.model import ModelTrainer

if __name__ == '__main__':
    try:
        logging.info('Data Injestion Started')
        obj = DataIngestion()
        train_data_path, test_data_path = obj.initiate_data_ingestion()
        logging.info("Data Ingestion Completed")

        logging.info("Data Transformation Started")
        data_transformation = Data_Transformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        logging.info("Data Ingestion Completed")

        logging.info('Model Training Started')
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_training(train_arr, test_arr)
        logging.info("Model Training Completed")

        print('Completed')

    except Exception as e:
        logging.info("Error occurred In training pipeline")
        raise CustomException(e, sys)

# I have to fix the path location of logs when training-pipeline is run the logsfolder new is created in pipelines
# it doenot go in the originol one that is after the artifacts folder
