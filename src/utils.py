"""
Utility functions for the ML project
"""
import os
import pickle
import sys
from typing import Any, Dict, Tuple

from src.exception import CustomException
from src.logger import logging


def save_object(file_path: str, obj: Any) -> None:
    """
    Save a Python object to a pickle file
    
    Args:
        file_path (str): Path where to save the object
        obj (Any): Object to save
    
    Raises:
        CustomException: If saving fails
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        
        logging.info(f"Object saved successfully at {file_path}")
    
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str) -> Any:
    """
    Load a Python object from a pickle file
    
    Args:
        file_path (str): Path to the pickle file
    
    Returns:
        Any: The loaded object
    
    Raises:
        CustomException: If loading fails
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
        
        logging.info(f"Object loaded successfully from {file_path}")
        return obj
    
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(
    X_train, y_train, X_test, y_test, 
    models: Dict[str, Any],
    params: Dict[str, Dict[str, Any]]
) -> Dict[str, float]:
    """
    Evaluate multiple models and return their scores
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        models (Dict): Dictionary of models
        params (Dict): Dictionary of hyperparameters for each model
    
    Returns:
        Dict[str, float]: Model names and their test scores
    
    Raises:
        CustomException: If evaluation fails
    """
    try:
        report = {}
        
        for model_name, model in models.items():
            logging.info(f"Evaluating {model_name}")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Get score (assuming regression - adjust for classification)
            from sklearn.metrics import r2_score
            score = r2_score(y_test, y_pred)
            report[model_name] = score
            
            logging.info(f"{model_name} R2 Score: {score:.4f}")
        
        return report
    
    except Exception as e:
        raise CustomException(e, sys)
