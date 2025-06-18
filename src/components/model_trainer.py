import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evalute_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:

            logging.info("Splitting the training dataset and testing dataset")

            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbour Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(),
                "catboost Regressor": CatBoostRegressor(),
                "Adaboost Regressor": AdaBoostRegressor()
            }

            model_report :dict=evalute_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            ## To get the best model score
            best_model_score=max(sorted(model_report.values()))

            ## To get the best model name
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_models=models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model is found")
            
            logging.info(f"Best found model on training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_models
            )
            

            predicted=best_models.predict(X_test)
            r2_square=r2_score(y_test, predicted)
            return r2_square
        
        except Exception as e:
            raise CustomException(e, sys)
