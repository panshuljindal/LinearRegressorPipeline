import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from data_transformation import DataTransformation
from model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    trainDataPath: str = os.path.join('artifacts', 'train.csv')
    testDataPath: str = os.path.join('artifacts', 'test.csv')
    rawDataPath: str = os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestionConfig = DataIngestionConfig()
    
    def initiateDataIngestion(self):
        logging.info("Enter Data Ingestion Method")
        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info("Read dataset as dataframe")

            os.mkdir(os.path.dirname(self.ingestionConfig.rawDataPath)) if not os.path.exists(os.path.dirname(self.ingestionConfig.rawDataPath)) else None
            
            df.to_csv(self.ingestionConfig.rawDataPath, index=False)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestionConfig.trainDataPath, index=False)
            test_set.to_csv(self.ingestionConfig.testDataPath, index=False)

            logging.info("Data Ingestion Completed")

            return (
                self.ingestionConfig.trainDataPath,
                self.ingestionConfig.testDataPath
            )

        except Exception as e:
            logging.info("Error in Data Ingestion Method")
            raise CustomException(e,sys)

if __name__ == "__main__":
    dataIngestion = DataIngestion()
    trainData, testData = dataIngestion.initiateDataIngestion()

    dataTransformation = DataTransformation()
    trainArr, testArr, _ = dataTransformation.initiateDataTransformation(trainData, testData)

    modelTrainer = ModelTrainer()
    r2Score = modelTrainer.initiateModelTrainer(trainArr, testArr)

    print(r2Score)
