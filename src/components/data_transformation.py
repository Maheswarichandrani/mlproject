import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transform_object(self):
        '''
            1. Separate the numerical and categorical features in the dataset.
            2. Create individual pipelines for numerical and categorical features.
            3. Numerical pipeline: 
                  - Imputer (to handle missing values, e.g., using mean or median).
                  - StandardScaler (to scale the numerical values).
            4. Categorical pipeline: 
                  - Imputer (to handle missing values, e.g., using most frequent strategy).
                  - OneHotEncoder (to convert categories into numeric vectors).
                  - Optionally: Scaling (e.g., with StandardScaler or MinMaxScaler — **only if needed** and compatible).
            5. Use ColumnTransformer to combine both pipelines.
            6. ColumnTransformer applies each pipeline to its respective set of features.
            '''

        try:
            numerical_columns=['reading_score', 'writing_score']
            categorical_columns=[
                'gender', 
                'race_ethnicity', 
                'parental_level_of_education', 
                'lunch', 
                'test_preparation_course'
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoding", OneHotEncoder(handle_unknown='ignore')),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical features standard scaling has completed")
            logging.info("Categorical features encoding has completed.")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("categorical_pipeline", categorical_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test dataset has completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_object=self.get_data_transform_object()

            target_column_name="math_score"
            numerical_columns=["writing_score","reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Logging preprocessing object on training dataset and testing dataframe"
            )

            input_feature_train_arr=preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_object.transform(input_feature_test_df)

            #concatenates the two arrays
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info(f"Saved preprocessing objects")

            # to save the pickle file
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_object
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)