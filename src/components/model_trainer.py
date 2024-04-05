import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import evaluateModel, saveObject

@dataclass
class ModelTrainerConfig():
    trainedModelFilePath: str = os.path.join('artifacts', 'trained_model.pkl')

class ModelTrainer:
    def __init__(self):
        self.modelTrainerConfig = ModelTrainerConfig()

    def initiateModelTrainer(self, trainArray, testArray):
        try:
            logging.info("Splitting the data into train and test")
            X_train, y_train, X_test, y_test = (
                trainArray[:, :-1],
                trainArray[:, -1],
                testArray[:, :-1],
                testArray[:, -1],
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            modelReport = evaluateModel(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            bestModelScore = max(sorted(modelReport.values()))


            bestModelName = list(modelReport.keys())[
                list(modelReport.values()).index(bestModelScore)
            ]
            bestModel = models[bestModelName]

            if bestModelScore < 0.6:
                logging.error("No best model found. Best model score is less than 0.6")
                raise CustomException("No best model found", sys)
            
            logging.info(f"Best model found is {bestModelName} with score {bestModelScore}")

            saveObject(self.modelTrainerConfig.trainedModelFilePath, bestModel)

            predicted = bestModel.predict(X_test)

            r2Score = r2_score(y_test, predicted)

            return r2Score

        except Exception as e:
            raise CustomException(e, sys)





