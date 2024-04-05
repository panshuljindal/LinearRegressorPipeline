import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.utils import loadObject

class PredictPipeline():
    def __init__(self):
        pass

    def predict(self, features):
        try:
            modelPath = 'artifacts/trained_model.pkl'
            preprocessorPath = 'artifacts/preprocessor.pkl'
            model = loadObject(file_path=modelPath)
            preprocessor = loadObject(file_path=preprocessorPath)
            processedData = preprocessor.transform(features)
            prediction = model.predict(processedData)
            return prediction
        except Exception as e:
            raise CustomException(e, sys)


class CustomData():
    def __init__(
        self,
        gender: str,
        race_ethnicity: int,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):
        
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
    
    def getDataAsDataFrame(self):
        try:
            customDataInput= {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }
            return pd.DataFrame(customDataInput)
        
        except Exception as e:
            raise CustomException(e, sys)
