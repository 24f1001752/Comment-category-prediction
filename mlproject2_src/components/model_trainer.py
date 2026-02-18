import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from mlproject2_src.exception import CustomException
from mlproject2_src.logger import logging
from mlproject2_src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "best_comment_classifier.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,X,Y):
        try:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, Y, test_size=0.2, random_state=42, stratify=Y)

            model = LogisticRegression(
                max_iter=2000,
                n_jobs=-1,
                class_weight="balanced",
                solver="lbfgs",
                multi_class="auto",
            )

            model.fit(X_tr, y_tr)

            y_pred = model.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average="macro")

            logging.info(f"Validation Accuracy: {acc}")
            logging.info(f"Validation Macro-F1: {f1}")

            save_object(self.model_trainer_config.trained_model_file_path, model)
            logging.info(f"Saved model: {self.model_trainer_config.trained_model_file_path}")

            return {"val_accuracy": float(acc), "val_macro_f1": float(f1)}

        except Exception as e:
            raise CustomException(e, sys)
