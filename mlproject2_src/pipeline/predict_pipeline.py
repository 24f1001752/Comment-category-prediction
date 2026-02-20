import sys
import os
import pandas as pd

from mlproject2_src.exception import CustomException
from mlproject2_src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame, return_label_names: bool = True):
        """
        features: DataFrame with at least 'comment' column.
                 Optional numeric columns: upvote, downvote, emoticon_1..3, if_1, if_2, race, religion, gender, disability
        return_label_names: if True, returns original label strings using label_encoder.pkl (if available)
        """
        try:
            model_path = os.path.join("artifacts", "best_comment_classifier.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            label_encoder_path = os.path.join("artifacts", "label_encoder.pkl")

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            X = preprocessor.transform(features)
            pred_encoded = model.predict(X)

            if return_label_names and os.path.exists(label_encoder_path):
                le = load_object(label_encoder_path)
                return le.inverse_transform(pred_encoded)

            return pred_encoded

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Single-row input builder for your comment dataset.
    Provide what you have; anything not provided defaults to 0/empty.
    """
    def __init__(
        self,
        comment: str,
        upvote: int = 0,
        downvote: int = 0,
        emoticon_1: int = 0,
        emoticon_2: int = 0,
        emoticon_3: int = 0,
        if_1: int = 0,
        if_2: int = 0,
        race: int = 0,
        religion: int = 0,
        gender: int = 0,
        disability: int = 0,
    ):
        self.comment = comment
        self.upvote = upvote
        self.downvote = downvote
        self.emoticon_1 = emoticon_1
        self.emoticon_2 = emoticon_2
        self.emoticon_3 = emoticon_3
        self.if_1 = if_1
        self.if_2 = if_2
        self.race = race
        self.religion = religion
        self.gender = gender
        self.disability = disability

    def get_data_as_data_frame(self) -> pd.DataFrame:
        try:
            custom_data_input_dict = {
                "comment": [self.comment],
                "upvote": [self.upvote],
                "downvote": [self.downvote],
                "emoticon_1": [self.emoticon_1],
                "emoticon_2": [self.emoticon_2],
                "emoticon_3": [self.emoticon_3],
                "if_1": [self.if_1],
                "if_2": [self.if_2],
                "race": [self.race],
                "religion": [self.religion],
                "gender": [self.gender],
                "disability": [self.disability],
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
