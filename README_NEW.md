# ML Project

A complete machine learning project with automated data ingestion, transformation, model training, and a Flask web application for predictions.

## 📁 Project Structure

```
practice/
├── src/
│   ├── components/
│   │   ├── data_ingestion.py       # Data loading and splitting
│   │   ├── data_transformation.py  # Feature engineering
│   │   ├── data_evaluation.py      # Model evaluation metrics
│   │   └── model_trainer.py        # Model training
│   ├── pipeline/
│   │   ├── train_pipeline.py       # Training orchestration
│   │   └── predict_pipeline.py     # Prediction handling
│   ├── exception.py                # Custom exception handling
│   ├── logger.py                   # Logging configuration
│   └── utils.py                    # Utility functions
├── templates/                      # HTML templates for Flask app
├── artifacts/                      # Trained models and preprocessors
├── logs/                          # Application logs
├── app.py                         # Flask web application
└── requirements.txt               # Python dependencies

```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/aksh-ay06/practice.git
cd practice
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Training the Model

Run the training pipeline:

```bash
python -m src.pipeline.train_pipeline
```

This will:
1. Ingest and split data (80/20 train/test)
2. Apply transformations (imputation, scaling, encoding)
3. Train multiple models (Linear Regression, Random Forest, Gradient Boosting)
4. Select and save the best model
5. Generate evaluation metrics

## 🌐 Running the Web Application

Start the Flask server:

```bash
python app.py
```

The app will be available at `http://localhost:5000`

### Web Interface Routes

- `/` - Home page
- `/predict` - Prediction form (GET/POST)
- `/health` - Health check endpoint
- `/api/predict` - API endpoint for JSON predictions
- `/train` - Trigger model training via API

### API Usage

**Make a prediction via API:**

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "feature1": 10.5,
    "feature2": 20.3,
    "feature3": 30.1
  }'
```

**Response:**
```json
{
  "prediction": 42.1234,
  "status": "success"
}
```

## 📝 Features

✅ **Automated ML Pipeline**
- Data ingestion with train/test split
- Automatic feature type detection
- Preprocessing pipelines (numerical & categorical)
- Multi-model training and evaluation

✅ **Web Application**
- Beautiful UI for predictions
- REST API endpoints
- Health monitoring
- Error handling with custom pages

✅ **Best Practices**
- Type hints throughout
- Comprehensive logging
- Exception handling
- Modular architecture
- Configuration management with dataclasses

## 🛠️ Technologies Used

- **Python** - Core programming language
- **Flask** - Web framework
- **scikit-learn** - Machine learning
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **XGBoost** - Gradient boosting
- **CatBoost** - Categorical boosting

## 📈 Model Metrics

The system evaluates models using:
- **Regression**: R², RMSE, MAE
- **Classification**: F1-Score, ROC-AUC, Confusion Matrix

## 🔧 Configuration

Update feature names in:
- `app.py` - Update form fields in CustomData
- `templates/predict.html` - Update HTML form inputs
- `src/components/data_transformation.py` - Update target column name

## 📦 Artifacts

Trained models and preprocessors are saved in the `artifacts/` directory:
- `model.pkl` - Best trained model
- `preprocessor.pkl` - Feature transformation pipeline
- `train.csv` / `test.csv` - Processed datasets
- `data.csv` - Raw data

## 📋 Logs

Application logs are stored in the `logs/` directory with timestamps.

## 👤 Author

**Akshay Patel**
- Email: ap00143@mix.wvu.edu
- GitHub: [@aksh-ay06](https://github.com/aksh-ay06)

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 🙏 Acknowledgments

- Built following machine learning best practices
- Inspired by end-to-end ML project patterns
- Based on BRFSS 2023 analysis patterns
