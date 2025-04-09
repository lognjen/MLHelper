# MLHelper

A Python toolkit for streamlining machine learning workflows, from data preprocessing to model evaluation.

## Features

- **Data Preprocessing**: Handle missing values, encode categorical variables, scale numerical features, and identify outliers
- **Feature Engineering**: Create polynomial features, interaction terms, bin numerical features, and extract date features
- **Model Evaluation**: Evaluate model performance with various metrics and cross-validation techniques
- **Visualization**: Visualize feature importance, correlations, distributions, and model performance
- **Model Helpers**: Split data, perform hyperparameter tuning, and save/load models

## Installation

```bash
pip install git+https://github.com/lognjen/MLHelper.git
```

## Module Overview

- `data_processing.py` - Contains the DataPreprocessor class with methods for handling missing values, encoding, scaling, and removing outliers
- `feature_engineering.py` - Contains the FeatureEngineering class with methods for polynomial features, interaction terms, binning, and date features
- `model_evaluation.py` - Contains the ModelEvaluation class with methods for evaluating models and cross-validation
- `visualization.py` - Contains the Visualization class with methods for plotting feature importance, correlations, and distributions
- `model_helper.py` - Contains the ModelHelper class with methods for data splitting, grid search, and model persistence

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## License

This project is licensed under the MIT License - see the LICENSE file for details.
