"""
Data transformation module - handles preprocessing and feature engineering
"""
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationConfig:
    """Configuration for data transformation"""
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    """
    Handle data transformation including imputation, encoding, and scaling
    """
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(
        self, 
        numerical_cols: List[str], 
        categorical_cols: List[str]
    ) -> ColumnTransformer:
        """
        Create preprocessing pipelines for numerical and categorical features
        
        Args:
            numerical_cols (List[str]): List of numerical column names
            categorical_cols (List[str]): List of categorical column names
        
        Returns:
            ColumnTransformer: Fitted preprocessor pipeline
        
        Raises:
            CustomException: If pipeline creation fails
        """
        try:
            # Numerical pipeline: imputation → scaling
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            
            # Categorical pipeline: imputation → one-hot encoding
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore"))
                ]
            )
            
            logging.info(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")
            logging.info(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
            
            # Combine pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_cols),
                    ("cat", cat_pipeline, categorical_cols)
                ]
            )
            
            logging.info("Data Transformer object created successfully")
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(
        self, 
        train_path: str, 
        test_path: str, 
        target_column: str = "target"
    ) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, str]:
        """
        Apply transformations to train and test data
        
        Args:
            train_path (str): Path to training data
            test_path (str): Path to test data
            target_column (str): Name of target column
        
        Returns:
            Tuple: Transformed train features, test features, train target, test target, preprocessor path
        
        Raises:
            CustomException: If transformation fails
        """
        logging.info("Data Transformation initiated")
        
        try:
            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info(f"Train data shape: {train_df.shape}")
            logging.info(f"Test data shape: {test_df.shape}")
            
            # Separate features and target
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]
            
            logging.info("Features and target separated")
            
            # Identify numerical and categorical columns
            numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
            
            logging.info(f"Found {len(numerical_cols)} numerical and {len(categorical_cols)} categorical columns")
            
            # Get preprocessor object
            preprocessor = self.get_data_transformer_object(numerical_cols, categorical_cols)
            
            # Fit and transform train data
            X_train_transformed = preprocessor.fit_transform(X_train)
            logging.info(f"Train data transformation completed. Shape: {X_train_transformed.shape}")
            
            # Transform test data
            X_test_transformed = preprocessor.transform(X_test)
            logging.info(f"Test data transformation completed. Shape: {X_test_transformed.shape}")
            
            # Create artifacts directory
            os.makedirs(os.path.dirname(self.transformation_config.preprocessor_obj_file_path), exist_ok=True)
            
            # Save preprocessor object
            with open(self.transformation_config.preprocessor_obj_file_path, 'wb') as f:
                pickle.dump(preprocessor, f)
            
            logging.info(f"Preprocessor saved at {self.transformation_config.preprocessor_obj_file_path}")
            
            return (
                X_train_transformed,
                X_test_transformed,
                y_train,
                y_test,
                self.transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    transformation = DataTransformation()
    X_train, X_test, y_train, y_test, preprocessor_path = transformation.initiate_data_transformation(
        train_path='artifacts/train.csv',
        test_path='artifacts/test.csv'
    )
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Preprocessor saved at: {preprocessor_path}")