import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")  # Fixed typo

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    
    def get_data_transformer_object(self):
        try:
            
            preprocessor = ColumnTransformer([
                ("tfidf", TfidfVectorizer(max_features=2000), "comment"),
                ("num", StandardScaler(), ["upvote", "downvote", "if_1", "if_2"]),
            ], remainder='drop')  

            logging.info("Simple preprocessor: TF-IDF + numeric scaling")
            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)

        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info(f"Train shape: {train_df.shape}")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            # Target column for comment project
            target_column_name = "label"

            # Train data: features + target
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Test data: only features (no target)
            input_feature_test_df = test_df

            logging.info("Applying preprocessing object")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Encode target labels
            label_encoder = LabelEncoder()
            target_feature_train_arr = label_encoder.fit_transform(target_feature_train_df)

            # Return arrays for model trainer (X_train+y_train, X_test)
            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = input_feature_test_arr

            logging.info("Saved preprocessing object and label encoder")

            # Save preprocessor
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # #Save label encoder
            save_object(
                file_path=os.path.join('artifacts', 'label_encoder.pkl'),
                obj=label_encoder
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
