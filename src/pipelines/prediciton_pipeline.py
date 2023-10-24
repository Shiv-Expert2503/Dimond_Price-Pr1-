import sys, os
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def prediction(self, features):
        try:
            logging.info("Prediction Started")
            preprocessor_path = (os.path.join(os.getcwd(), 'artifacts', 'preprocessor.pkl'))
            model_path = (os.path.join(os.getcwd(), 'artifacts', 'model.pkl'))
            logging.info(f"FilePAth at Prediction of preprocessor {preprocessor_path}")
            logging.info(f"FilePAth at Prediction of model {model_path}")
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)
            logging.info("Prediction Completed")
            return pred

        except Exception as e:
            logging.info("Error Occurred while predicting")
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 carat: float,
                 depth: float,
                 table: float,
                 x: float,
                 y: float,
                 z: float,
                 cut: str,
                 color: str,
                 clarity: str):

        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity

    def get_data_as_dataframe(self):
        try:
            logging.info("Inputed Data has started making into the data Frame")
            custom_data_input_dict = {
                'carat': [self.carat],
                'depth': [self.depth],
                'table': [self.table],
                'x': [self.x],
                'y': [self.y],
                'z': [self.z],
                'cut': [self.cut],
                'color': [self.color],
                'clarity': [self.clarity]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in While Gatherring External Data')
            raise CustomException(e, sys)
# import os
# print((os.path.join(os.getcwd(),'artifacts', 'preprocessor.pkl')))