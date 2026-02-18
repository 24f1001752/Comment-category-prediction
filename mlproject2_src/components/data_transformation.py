import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, FunctionTransformer

from mlproject2_src.exception import CustomException
from mlproject2_src.logger import logging
from mlproject2_src.utils import save_object

##remove the lambda and use a top-level function
import numpy as np

def to_1d_str(x):
    return np.asarray(x).ravel().astype(str)



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")
    label_encoder_file_path: str = os.path.join("artifacts", "label_encoder.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, numeric_features):
        try:
            # IMPORTANT: TfidfVectorizer needs 1D iterable of strings. We force 2D -> 1D via ravel().
            text_pipeline = Pipeline(steps=[
                ("to_1d", FunctionTransformer(to_1d_str, validate=False)),
                ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
            ])

            # Numeric pipeline (works with sparse output; with_mean=False is safe)
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("scaler", StandardScaler(with_mean=False)),
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("text", text_pipeline, ["comment"]),
                    ("num", num_pipeline, numeric_features),
                ],
                remainder="drop",
                sparse_threshold=0.3,
            )

            logging.info("Created preprocessor: TF-IDF(comment) + numeric pipeline")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info(f"Train shape: {train_df.shape}")
            logging.info(f"Test shape: {test_df.shape}")

            target_column_name = "label"
            if target_column_name not in train_df.columns:
                raise ValueError("train.csv must contain 'label' column")

            if "comment" not in train_df.columns or "comment" not in test_df.columns:
                raise ValueError("Both train.csv and test.csv must contain 'comment' column")

            candidate_numeric = [
                "upvote", "downvote",
                "emoticon_1", "emoticon_2", "emoticon_3",
                "if_1", "if_2",
                "race", "religion", "gender", "disability",
            ]
            numeric_features = [c for c in candidate_numeric if c in train_df.columns and c in test_df.columns]
            



            preprocessing_obj = self.get_data_transformer_object(numeric_features=numeric_features)

            X_train_df = train_df.drop(columns=[target_column_name], axis=1)
            y_train = train_df[target_column_name]
            X_test_df = test_df.copy()
            for col in numeric_features:
                X_train_df[col] = pd.to_numeric(X_train_df[col], errors="coerce")
                X_test_df[col] = pd.to_numeric(X_test_df[col], errors="coerce")


            logging.info("Applying preprocessing")
            X_train = preprocessing_obj.fit_transform(X_train_df)
            X_test = preprocessing_obj.transform(X_test_df)

            label_encoder = LabelEncoder()
            y_train_enc = label_encoder.fit_transform(y_train)

            # Fit/transform
            X_train = preprocessing_obj.fit_transform(X_train_df)
            X_test = preprocessing_obj.transform(X_test_df)

            label_encoder = LabelEncoder()
            y_train_enc = label_encoder.fit_transform(y_train)

            save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessing_obj)
            save_object(self.data_transformation_config.label_encoder_file_path, label_encoder)

            return X_train, y_train_enc, X_test, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
