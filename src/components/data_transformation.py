from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import sys, os
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


#Data Transformation config
@dataclass
class Data_transformation_config:
    preprocessor_obj_file_path = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'artifacts', 'preprocessor.pkl'))


#Data Transformation class
class Data_Transformation:
    def __init__(self):
        self.data_transformation_config = Data_transformation_config()

    def get_data_transformation_object(self):

        try:
            logging.info('Data Tranformation Started')

            cat_col = ['cut', 'color', 'clarity']
            num_col = ['carat', 'depth', 'table', 'x', 'y', 'z']

            cut_cat = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
            col_cat = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            cla_cat = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

            logging.info('Pipeline initiated')

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder', OrdinalEncoder(categories=[cut_cat, col_cat, cla_cat])),
                    ('scaler', StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, num_col),
                ('cat_pipeline', cat_pipeline, cat_col)
            ])

            logging.info('Pipeline is completed')

            return preprocessor

        except Exception as e:
            logging.info("Error occurred in Data Transformation")
            raise CustomException(e, sys)


    def initiate_data_transformation(self,train_path,test_path):

        try:

            logging.info("Reading data")
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Reading Data Completed")

            logging.info(f"Train Dataframe head : \n{train_df.head().to_string()}")
            logging.info(f"Test Dataframe head : \n{test_df.head().to_string()}")

            logging.info('Obtaining Preprocessing Object')

            preprocessor_obj=self.get_data_transformation_object()

            target_column_name='price'
            drop_columns=[target_column_name,'id']

            input_feature_train_df=train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            logging.info('Dependent and independent features are seperated')

            logging.info("Applying the transformation")

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            logging.info("Preprocessor pickle is created and saved")

            return(

                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            logging.info('Error occurred while initiating data transformation')
            raise CustomException(e,sys)

