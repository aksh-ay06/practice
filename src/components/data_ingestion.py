import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion"""
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


class DataIngestion:
    """
    Handle data ingestion from raw source to train/test split
    """
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self) -> Tuple[str, str]:
        """
        Ingest data and split into train and test sets
        
        Returns:
            Tuple[str, str]: Paths to train and test data files
        
        Raises:
            CustomException: If data ingestion fails
        """
        logging.info("Data Ingestion method starts")
        
        try:
            # Read the dataset (update path to your actual data source)
            df = pd.read_csv('data/data.csv')
            logging.info(f"Dataset read successfully: {df.shape[0]} rows × {df.shape[1]} cols")
            
            # Create artifacts directory
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            
            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved at {self.ingestion_config.raw_data_path}")
            
            # Split into train and test sets
            logging.info("Initiating train-test split (80-20 split)")
            train_set, test_set = train_test_split(
                df, 
                test_size=0.2, 
                random_state=42
            )
            
            # Save train and test data
            train_set.to_csv(
                self.ingestion_config.train_data_path, 
                index=False, 
                header=True
            )
            test_set.to_csv(
                self.ingestion_config.test_data_path, 
                index=False, 
                header=True
            )
            
            logging.info(
                f"Train data saved: {train_set.shape[0]} rows at {self.ingestion_config.train_data_path}"
            )
            logging.info(
                f"Test data saved: {test_set.shape[0]} rows at {self.ingestion_config.test_data_path}"
            )
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()