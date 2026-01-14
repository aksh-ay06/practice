"""
Model training module - handles training and evaluation of ML models
"""
import os
import sys
from dataclasses import dataclass
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle

from src.exception import CustomException
from src.logger import logging


@dataclass
class ModelTrainerConfig:
    """Configuration for model training"""
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    """
    Train machine learning models and evaluate performance
    """
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(
        self, 
        X_train: np.ndarray, 
        X_test: np.ndarray, 
        y_train: np.ndarray, 
        y_test: np.ndarray
    ) -> Tuple[Any, Dict[str, Dict[str, float]]]:
        """
        Train multiple models and return the best performing one
        
        Args:
            X_train (np.ndarray): Training features
            X_test (np.ndarray): Test features
            y_train (np.ndarray): Training target
            y_test (np.ndarray): Test target
        
        Returns:
            Tuple: Best model and report of all models' performance
        
        Raises:
            CustomException: If model training fails
        """
        logging.info("Model Training initiated")
        
        try:
            # Define models
            models = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
            }
            
            model_report = {}
            
            # Train and evaluate each model
            for model_name, model in models.items():
                logging.info(f"Training {model_name}")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                # Calculate metrics
                train_mse = mean_squared_error(y_train, y_train_pred)
                test_mse = mean_squared_error(y_test, y_test_pred)
                train_rmse = train_mse ** 0.5
                test_rmse = test_mse ** 0.5
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                train_mae = mean_absolute_error(y_train, y_train_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)
                
                model_report[model_name] = {
                    'model': model,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_mae': train_mae,
                    'test_mae': test_mae
                }
                
                logging.info(
                    f"{model_name} → Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}, "
                    f"Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}"
                )
            
            # Find best model based on test R2 score
            best_model_name = max(model_report, key=lambda x: model_report[x]['test_r2'])
            best_model = model_report[best_model_name]['model']
            
            logging.info(f"Best model: {best_model_name}")
            logging.info(f"Best model Test R2 Score: {model_report[best_model_name]['test_r2']:.4f}")
            logging.info(f"Best model Test RMSE: {model_report[best_model_name]['test_rmse']:.4f}")
            
            # Create artifacts directory
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            
            # Save best model
            with open(self.model_trainer_config.trained_model_file_path, 'wb') as f:
                pickle.dump(best_model, f)
            
            logging.info(f"Best model saved at {self.model_trainer_config.trained_model_file_path}")
            
            return best_model, model_report
        
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    trainer = ModelTrainer()
    # X_train, X_test, y_train, y_test should come from data_transformation
    # best_model, report = trainer.initiate_model_trainer(X_train, X_test, y_train, y_test)