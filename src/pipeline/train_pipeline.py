"""
Training pipeline - orchestrates the entire ML workflow
"""
import os
import sys
from typing import Tuple

import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.data_evaluation import DataEvaluation
from src.exception import CustomException
from src.logger import logging


class TrainingPipeline:
    """
    End-to-end training pipeline that orchestrates:
    1. Data ingestion
    2. Data transformation
    3. Model training
    4. Model evaluation
    """
    
    def __init__(self):
        """Initialize all pipeline components"""
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()
        self.data_evaluation = DataEvaluation()
    
    def run(self) -> Tuple[str, dict]:
        """
        Execute the complete training pipeline
        
        Returns:
            Tuple[str, dict]: Path to trained model and evaluation report
        
        Raises:
            CustomException: If any step in the pipeline fails
        """
        try:
            logging.info("=" * 50)
            logging.info("Training Pipeline Started")
            logging.info("=" * 50)
            
            # Step 1: Data Ingestion
            logging.info("\nStep 1: Data Ingestion")
            logging.info("-" * 50)
            train_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion()
            logging.info(f"Train data path: {train_data_path}")
            logging.info(f"Test data path: {test_data_path}")
            
            # Step 2: Data Transformation
            logging.info("\nStep 2: Data Transformation")
            logging.info("-" * 50)
            X_train, X_test, y_train, y_test, preprocessor_path = (
                self.data_transformation.initiate_data_transformation(
                    train_data_path, 
                    test_data_path
                )
            )
            logging.info(f"Train shape: {X_train.shape}")
            logging.info(f"Test shape: {X_test.shape}")
            logging.info(f"Preprocessor saved at: {preprocessor_path}")
            
            # Step 3: Model Training
            logging.info("\nStep 3: Model Training")
            logging.info("-" * 50)
            best_model, model_report = self.model_trainer.initiate_model_trainer(
                X_train, X_test, y_train, y_test
            )
            logging.info("Model training completed")
            
            # Step 4: Model Evaluation
            logging.info("\nStep 4: Model Evaluation")
            logging.info("-" * 50)
            y_pred = best_model.predict(X_test)
            evaluation_metrics = self.data_evaluation.evaluate_regression(
                y_test, y_pred, model_name="Best Model"
            )
            self.data_evaluation.save_evaluation_report(evaluation_metrics)
            
            logging.info("\n" + "=" * 50)
            logging.info("Training Pipeline Completed Successfully")
            logging.info("=" * 50)
            
            model_path = self.model_trainer.model_trainer_config.trained_model_file_path
            return model_path, evaluation_metrics
        
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    model_path, metrics = pipeline.run()
    print(f"\n✓ Model saved at: {model_path}")
    print(f"✓ Metrics: {metrics}")
