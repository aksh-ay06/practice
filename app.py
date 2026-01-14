"""
Flask web application for ML model predictions
"""
import os
import sys
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.pipeline.predict_pipeline import PredictionPipeline, CustomData

# Initialize Flask app
app = Flask(__name__)

# Initialize prediction pipeline
try:
    pipeline = PredictionPipeline()
    logging.info("Prediction pipeline initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize prediction pipeline: {str(e)}")
    pipeline = None


@app.route('/')
def index():
    """
    Render the home page
    """
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    """
    Handle prediction requests
    """
    if request.method == 'GET':
        return render_template('predict.html')
    
    else:
        try:
            # Get data from form
            # Adjust these fields based on your actual features
            data = CustomData(
                feature1=float(request.form.get('feature1')),
                feature2=float(request.form.get('feature2')),
                feature3=float(request.form.get('feature3')),
                # Add more features as needed
            )
            
            logging.info(f"Received prediction request with data: {request.form.to_dict()}")
            
            # Get prediction
            if pipeline is None:
                raise ValueError("Prediction pipeline not initialized")
            
            pred_df = data.get_data_as_dataframe()
            predictions = pipeline.predict(pred_df)
            
            result = float(predictions[0])
            logging.info(f"Prediction result: {result}")
            
            return render_template('predict.html', results=result)
        
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            raise CustomException(e, sys)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint for predictions (JSON)
    """
    try:
        # Get JSON data
        json_data = request.get_json()
        
        if not json_data:
            return jsonify({'error': 'No data provided'}), 400
        
        logging.info(f"API prediction request: {json_data}")
        
        # Create CustomData object
        data = CustomData(**json_data)
        
        # Get prediction
        if pipeline is None:
            raise ValueError("Prediction pipeline not initialized")
        
        pred_df = data.get_data_as_dataframe()
        predictions = pipeline.predict(pred_df)
        
        result = float(predictions[0])
        logging.info(f"API prediction result: {result}")
        
        return jsonify({
            'prediction': result,
            'status': 'success'
        })
    
    except Exception as e:
        logging.error(f"API prediction error: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/health')
def health_check():
    """
    Health check endpoint
    """
    try:
        model_loaded = pipeline is not None and pipeline.model is not None
        return jsonify({
            'status': 'healthy',
            'model_loaded': model_loaded
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


@app.route('/train', methods=['POST'])
def trigger_training():
    """
    Endpoint to trigger model training
    """
    try:
        from src.pipeline.train_pipeline import TrainingPipeline
        
        logging.info("Training pipeline triggered via API")
        
        training_pipeline = TrainingPipeline()
        model_path, metrics = training_pipeline.run()
        
        return jsonify({
            'status': 'success',
            'message': 'Training completed successfully',
            'model_path': model_path,
            'metrics': metrics
        })
    
    except Exception as e:
        logging.error(f"Training error: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logging.error(f"Internal server error: {str(error)}")
    return render_template('500.html'), 500


if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
