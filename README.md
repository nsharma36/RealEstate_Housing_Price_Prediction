# House Price Prediction Project

A machine learning project that predicts housing prices using the Boston Housing Dataset. This project demonstrates the complete machine learning workflow from data exploration and preprocessing to model training and evaluation.

## Project Overview

**Real Estate Price Predictor** uses regression models to predict median home values based on various features such as crime rate, number of rooms, distance to employment centers, and other socioeconomic factors.

### Dataset Information
- **Name:** Boston Housing Data
- **Size:** 506 instances
- **Features:** 13 continuous attributes + 1 target variable
- **Target Variable:** MEDV (Median value of owner-occupied homes in $1,000's)

### Dataset Attributes

1. **CRIM** - Per capita crime rate by town
2. **ZN** - Proportion of residential land zoned for lots over 25,000 sq.ft.
3. **INDUS** - Proportion of non-retail business acres per town
4. **CHAS** - Charles River dummy variable (1 if bounds river; 0 otherwise)
5. **NOX** - Nitric oxides concentration (parts per 10 million)
6. **RM** - Average number of rooms per dwelling
7. **AGE** - Proportion of owner-occupied units built prior to 1940
8. **DIS** - Weighted distances to five Boston employment centres
9. **RAD** - Index of accessibility to radial highways
10. **TAX** - Full-value property-tax rate per $10,000
11. **PTRATIO** - Pupil-teacher ratio by town
12. **B** - 1000(Bk - 0.63)² where Bk is the proportion of blacks by town
13. **LSTAT** - % lower status of the population

## Project Structure

```
ProjectHousePricePrediction/
├── README.md                          # Project documentation
├── HousePricePredict.ipynb           # Main model training notebook
├── Model_Usage.ipynb                 # Notebook for using the trained model
├── HousePricePredict.joblib          # Serialized trained model
├── housing.csv                       # Housing dataset (CSV format)
├── housing.data                      # Housing dataset (raw data format)
├── housing.names                     # Dataset description and attribute information
└── ModelOutputs from Different Models/ # Directory containing results from various models
```

## Files Description

- **HousePricePredict.ipynb** - Main Jupyter notebook containing:
  - Data exploration and analysis
  - Data preprocessing and feature engineering
  - Train-test split with stratified sampling
  - Model training and hyperparameter tuning
  - Model evaluation metrics

- **Model_Usage.ipynb** - Demonstration notebook showing:
  - How to load the pre-trained model
  - Making predictions on new data
  - Example predictions

- **HousePricePredict.joblib** - Serialized machine learning model ready for deployment

## Dependencies

- pandas - Data manipulation and analysis
- numpy - Numerical computing
- scikit-learn - Machine learning library
- joblib - Model serialization

Install dependencies with:
```bash
pip install pandas numpy scikit-learn joblib matplotlib seaborn
```

## Machine Learning Workflow

### 1. Data Loading & Exploration
- Load housing dataset using pandas
- Explore data types and missing values
- Analyze categorical and numerical attributes

### 2. Data Preprocessing
- **Train-Test Split:** 80-20 split for training and testing
- **Stratified Sampling:** Ensures balanced distribution of categorical variables (CHAS attribute)
- **Handling Missing Values:** Options include:
  - Removing rows with missing values
  - Dropping entire attributes
  - Imputing with mean/median/mode values

### 3. Feature Engineering
- Attribute scaling and normalization
- Creating additional features from existing ones

### 4. Model Training
- Multiple regression algorithms evaluated
- Models are compared on various metrics
- Results stored in "ModelOutputs from Different Models" directory

### 5. Model Evaluation
- Metrics used: MSE (Mean Squared Error), R² Score
- Cross-validation for robust performance estimation

## Usage

### Training the Model

Open `HousePricePredict.ipynb` in Jupyter Notebook:
```bash
jupyter notebook HousePricePredict.ipynb
```

Run all cells to:
1. Load and explore the data
2. Preprocess the data
3. Train the machine learning model
4. Save the model to `HousePricePredict.joblib`

### Making Predictions

Use the pre-trained model in `Model_Usage.ipynb`:

```python
import numpy as np
from joblib import load

# Load the trained model
model = load('HousePricePredict.joblib')

# Prepare features (13 normalized features)
features = np.array([[-1.02731, 0.1, 7.07, -1.0, 0.469, 6.421, 78.9, 6.9671, 2.0, 242.0, 19.8, 396.90, 1.14]])

# Make prediction
prediction = model.predict(features)
print(f"Predicted house price: ${prediction[0] * 1000:.2f}")
```

## Model Performance

The final model achieves strong performance metrics on the test set. Detailed performance comparisons for different algorithms can be found in the `ModelOutputs from Different Models` directory.

## Key Features

✓ Complete ML pipeline implementation
✓ Stratified sampling for better data distribution
✓ Multiple model comparison and evaluation
✓ Pre-trained model ready for deployment
✓ Easy-to-use prediction interface
✓ Comprehensive data exploration and visualization

## Data Source

**Original Source:** StatLib library at Carnegie Mellon University
**Citation:** Harrison, D. and Rubinfeld, D.L. 'Hedonic prices and the demand for clean air', J. Environ. Economics & Management, vol.5, 81-102, 1978.

## Future Improvements

- Feature selection and optimization
- Ensemble methods for better predictions
- Deployment as REST API
- Web interface for predictions
- Time-series analysis if temporal data is available

## License

This project uses publicly available Boston Housing Dataset for educational purposes.

## Author


## Acknowledgments

- UCI Machine Learning Repository for the housing dataset
- scikit-learn for the excellent machine learning tools
