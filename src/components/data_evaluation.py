"""
Data evaluation and analysis module
"""
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, 
    r2_score, 
    mean_absolute_error,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    f1_score
)

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataEvaluationConfig:
    """Configuration for data evaluation"""
    evaluation_report_path: str = "artifacts/evaluation_report.txt"


class DataEvaluation:
    """
    Evaluate model performance using various metrics
    """
    def __init__(self):
        self.evaluation_config = DataEvaluationConfig()
    
    def evaluate_regression(self, y_true, y_pred, model_name: str = "Model") -> dict:
        """
        Evaluate regression model performance
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            model_name (str): Name of the model
        
        Returns:
            dict: Dictionary containing evaluation metrics
        
        Raises:
            CustomException: If evaluation fails
        """
        try:
            logging.info(f"Evaluating regression model: {model_name}")
            
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            metrics = {
                'model_name': model_name,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2_score': r2
            }
            
            logging.info(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
            
            return metrics
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def evaluate_classification(self, y_true, y_pred, y_proba=None, model_name: str = "Model") -> dict:
        """
        Evaluate classification model performance
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            y_proba: Probability predictions (optional)
            model_name (str): Name of the model
        
        Returns:
            dict: Dictionary containing evaluation metrics
        
        Raises:
            CustomException: If evaluation fails
        """
        try:
            logging.info(f"Evaluating classification model: {model_name}")
            
            f1 = f1_score(y_true, y_pred, average='weighted')
            cm = confusion_matrix(y_true, y_pred)
            
            metrics = {
                'model_name': model_name,
                'f1_score': f1,
                'confusion_matrix': cm
            }
            
            # Add ROC-AUC if probabilities provided and binary classification
            if y_proba is not None and len(np.unique(y_true)) == 2:
                roc_auc = roc_auc_score(y_true, y_proba)
                metrics['roc_auc'] = roc_auc
                logging.info(f"F1 Score: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
            else:
                logging.info(f"F1 Score: {f1:.4f}")
            
            logging.info(f"Classification Report:\n{classification_report(y_true, y_pred)}")
            
            return metrics
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def save_evaluation_report(self, metrics: dict) -> None:
        """
        Save evaluation report to file
        
        Args:
            metrics (dict): Dictionary containing evaluation metrics
        
        Raises:
            CustomException: If saving fails
        """
        try:
            import os
            os.makedirs(os.path.dirname(self.evaluation_config.evaluation_report_path), exist_ok=True)
            
            with open(self.evaluation_config.evaluation_report_path, 'w') as f:
                for key, value in metrics.items():
                    f.write(f"{key}: {value}\n")
            
            logging.info(f"Evaluation report saved at {self.evaluation_config.evaluation_report_path}")
        
        except Exception as e:
            raise CustomException(e, sys)
