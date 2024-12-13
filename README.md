# House Price Prediction Model

## Overview
This project implements a machine learning model to predict house prices using various features such as year built, number of bedrooms, amenities, and location data. The model uses Random Forest Regression and compares its performance with OLS (Ordinary Least Squares) and Decision Tree approaches.

## Data Description
The dataset contains housing data from 2016-2017 with the following key features:
- Property characteristics (bedrooms, bathrooms, square footage)
- Building information (year built, number of floors)
- Financial details (maintenance costs, taxes, common charges)
- Amenities (garage, pets allowed, fuel type)
- Property type (coop/condo)
- Kitchen and dining room specifications

## Technical Implementation
### Data Preprocessing
- Feature selection reducing to 25 key features
- Binary variable conversion to 0/1 dummies
- Min-Max scaling for year built feature
- Missing value imputation using Random Forest
- Categorical variable encoding

### Models Implemented
1. Random Forest Regressor (Primary Model)
2. OLS (Ordinary Least Squares) Regression
3. Decision Tree Regressor (for comparison)

### Key Features
- Automated missing value handling
- Feature importance analysis
- Model performance comparison
- Visualization of decision trees
- Price prediction functionality for new properties

## Model Performance
The project includes various performance metrics:
- R-squared score
- Root Mean Squared Error (RMSE)
- Training and testing set evaluations
- Feature importance rankings

## Requirements
```python
- pandas
- numpy
- statsmodels
- scikit-learn
- matplotlib
- graphviz
- PIL
```

## Usage
1. Load and preprocess the data:
```python
data_frame = pd.read_csv("housing_data_2016_2017.csv")
```

2. Train the model:
```python
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
```

3. Make predictions:
```python
predicted_price = rf_model.predict(new_house_df)
```

## File Structure
- `main.ipynb`: Main Jupyter notebook containing all analysis and model implementation
  - Data cleaning and preprocessing
  - Model training and evaluation
  - Visualization and analysis
  - Prediction functionality

## Future Improvements
- Feature engineering optimization
- Hyperparameter tuning
- Additional model architectures
- Cross-validation implementation
- Enhanced visualization features

## Contributors
This project is maintained by Carlos Vega.
