import os
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.utils import saveObject

@dataclass
class DataTransformationConfig:
    preprocessorObjectFilePath: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.dataTransformationConfig = DataTransformationConfig()
    
    def getDataTransformerObject(self):
        # This method will return the preprocessor object which will be used to transform the data.
        
        logging.info("Enter Data Transformation Method")
        try:
            numericalColumns = ['writing_score', 'reading_score']
            categoricalColumns = [
                'gender', 'race_ethnicity',
                'parental_level_of_education',
                'lunch', 'test_preparation_course'
            ]
            
            numericalPipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scalar", StandardScaler())
                ]
            )

            categoricalPipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder()),
                    ("scalar", StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical and Categorical Pipeline Created")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("numerical", numericalPipeline, numericalColumns),
                    ("categorical", categoricalPipeline, categoricalColumns)
                ]
            )
            return preprocessor
        except Exception as e:
            logging.info("Error in Data Transformation Method")
            raise CustomException(e,sys)
        
    def initiateDataTransformation(self, trainDataPath, testDataPath):
        try:
            train_df = pd.read_csv(trainDataPath)
            test_df = pd.read_csv(testDataPath)

            logging.info("Read train and test dataset as dataframe")

            logging.info("Obtaining Preprocessor Object")
            preprocessor = self.getDataTransformerObject()

            targetColumn = "math_score"
            numericalColumns = ['writing_score', 'reading_score']

            inputFeaturesTrain = train_df.drop(columns=[targetColumn], axis=1)
            targetTrain = train_df[targetColumn]

            inputFeaturesTest = test_df.drop(columns=[targetColumn], axis=1)
            targetTest = test_df[targetColumn]

            logging.info("Applying preprocessor object on Train and Test Data")

            inputFeaturesTrain = preprocessor.fit_transform(inputFeaturesTrain)
            inputFeaturesTest = preprocessor.transform(inputFeaturesTest)

            logging.info("Data Transformation Completed")

            trainArr = np.c_[inputFeaturesTrain, np.array(targetTrain)]
            testArr = np.c_[inputFeaturesTest, np.array(targetTest)]

            logging.info(f"Saved preprocessing object.")

            saveObject(file_path=self.dataTransformationConfig.preprocessorObjectFilePath, obj=preprocessor)

            return (
                trainArr,
                testArr,
                self.dataTransformationConfig.preprocessorObjectFilePath,
            )

        except Exception as e: 
            raise CustomException(e,sys)
        
