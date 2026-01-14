"""
Prediction pipeline - handles making predictions on new data
"""
import os
import sys
from typing import Any, Union

import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictionPipeline:
    """
    Pipeline for making predictions using a trained model and preprocessor
    """
    
    def __init__(self, model_path: str = "artifacts/model.pkl", 
                 preprocessor_path: str = "artifacts/preprocessor.pkl"):
        """
        Initialize prediction pipeline with model and preprocessor
        
        Args:
            model_path (str): Path to the trained model
            preprocessor_path (str): Path to the preprocessor object
        """
        try:
            self.model_path = model_path
            self.preprocessor_path = preprocessor_path
            
            # Load model and preprocessor
            if os.path.exists(model_path) and os.path.exists(preprocessor_path):
                self.model = load_object(model_path)
                self.preprocessor = load_object(preprocessor_path)
                logging.info("Model and preprocessor loaded successfully")
            else:
                logging.warning(f"Model or preprocessor not found at specified paths")
                self.model = None
                self.preprocessor = None
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def predict(self, features: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            features (Union[pd.DataFrame, np.ndarray]): Input features for prediction
        
        Returns:
            np.ndarray: Predictions
        
        Raises:
            CustomException: If prediction fails or model is not loaded
        """
        try:
            if self.model is None or self.preprocessor is None:
                raise ValueError("Model or preprocessor not loaded")
            
            # Convert to DataFrame if necessary
            if isinstance(features, np.ndarray):
                features = pd.DataFrame(features)
            
            logging.info(f"Making predictions on {len(features)} samples")
            
            # Preprocess features
            features_transformed = self.preprocessor.transform(features)
            
            # Make predictions
            predictions = self.model.predict(features_transformed)
            
            logging.info(f"Predictions completed. Shape: {predictions.shape}")
            
            return predictions
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def predict_proba(self, features: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Get probability predictions (for classification models)
        
        Args:
            features (Union[pd.DataFrame, np.ndarray]): Input features for prediction
        
        Returns:
            np.ndarray: Probability predictions
        
        Raises:
            CustomException: If prediction fails or model doesn't support probabilities
        """
        try:
            if self.model is None or self.preprocessor is None:
                raise ValueError("Model or preprocessor not loaded")
            
            if not hasattr(self.model, 'predict_proba'):
                raise ValueError("Model does not support probability predictions")
            
            # Convert to DataFrame if necessary
            if isinstance(features, np.ndarray):
                features = pd.DataFrame(features)
            
            logging.info(f"Getting probability predictions for {len(features)} samples")
            
            # Preprocess features
            features_transformed = self.preprocessor.transform(features)
            
            # Get probability predictions
            probabilities = self.model.predict_proba(features_transformed)
            
            logging.info(f"Probability predictions completed. Shape: {probabilities.shape}")
            
            return probabilities
        
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Class to handle custom input data for predictions
    """
    
    def __init__(self, **kwargs):
        """
        Initialize with custom features
        
        Args:
            **kwargs: Feature names and values
        """
        try:
            self.data = pd.DataFrame([kwargs])
            logging.info(f"Custom data created with columns: {list(kwargs.keys())}")
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_data_as_dataframe(self) -> pd.DataFrame:
        """
        Get data as DataFrame
        
        Returns:
            pd.DataFrame: The custom data
        """
        return self.data


if __name__ == "__main__":
    # Example usage
    try:
        # Initialize prediction pipeline
        pipeline = PredictionPipeline()
        
        # Create sample data
        sample_data = CustomData(
            feature1=10.5,
            feature2=20.3,
            feature3=30.1
        )
        
        # Make predictions
        if pipeline.model is not None:
            predictions = pipeline.predict(sample_data.get_data_as_dataframe())
            print(f"Predictions: {predictions}")
    
    except Exception as e:
        logging.error(f"Error in prediction pipeline: {str(e)}")
