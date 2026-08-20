# California Residential Property Close Price Prediction

## Project Overview

This project develops a machine learning model to predict the **ClosePrice**, or final sales price, of a single residential property in California based on the property's characteristics at the time of the query.

The model uses property characteristics and geographic information to estimate the final sale price of residential properties.

## Dataset Source

The dataset consists of historical real estate transaction data sourced from the **California Regional Multiple Listing Service (CRMLS)**.

The data contains monthly records from **May 2025 through June 2026**, which were combined into a single dataset before preprocessing.

The target variable is **`ClosePrice`**, which represents the final sales price of the property.

The main features used in the final models include:

- `LivingArea`
- `BedroomsTotal`
- `BathroomsTotalInteger`
- `LotSizeArea`
- `Age`
- `Latitude`
- `Longitude`
- City
- County
- Unified school district

## Data Preprocessing

### Property Filtering

The dataset was restricted to properties where:

- `PropertyType = Residential`
- `PropertySubType = SingleFamilyResidence`

### Removing Invalid and Duplicate Data

The following observations were removed:

- Columns with more than 70% missing values
- Rows with missing or non-positive `ClosePrice`
- Duplicate rows
- Properties with non-positive `LivingArea` or `LotSizeArea`
- Observations with inconsistent bedroom, bathroom, and living-area values
- Observations with invalid latitude or longitude values

### Train/Test Split

The data was split chronologically using `CloseDate`.

The **most recent month was used as the test set**, while the **previous 12 months were used as the training set**. This approach better represents the real-world task of using historical transactions to predict prices for newer properties.

### Missing Values

Missing values were handled using information from the training data:

- Missing `AttachedGarageYN` values were replaced with the training-set mode, and a missingness indicator was created.
- Missing values in `LivingArea`, `LotSizeArea`, `BedroomsTotal`, and `BathroomsTotalInteger` were replaced with their respective training-set medians.
- Missing `City` and `CountyOrParish` values were replaced with `"Unknown"`.
- Missing `YearBuilt` values were replaced with the training-set median, and a `YearBuilt_missing` indicator was created.

A new **`Age`** feature was also created from `YearBuilt`.

### Geographic Features

California Unified School District boundary data was used to identify the school district associated with each property.

Latitude and longitude were converted into geographic points and spatially joined with the school district boundaries to obtain the corresponding `DistrictName`.

Latitude and longitude were also included directly as model features to capture geographic differences in property prices.

### Categorical Encoding

`DistrictName`, `City`, and `CountyOrParish` were converted into numerical features using **one-hot encoding**.

For City, cities with at least 50 observations in the training data were retained individually. Less frequent cities were grouped into `"Other"`.

The test set was then aligned with the training set so that both datasets contained the same feature columns.

### Target Transformation

Because `ClosePrice` is highly right-skewed, the target variable was log-transformed using:

`log1p(ClosePrice)`

for the Decision Tree and Random Forest models.

After making predictions, the values were converted back to the original price scale using `expm1()`.

## Models Tested

### Linear Regression

Linear Regression was used as a baseline model to provide a simple benchmark for comparison with more complex models.

### Decision Tree

A Decision Tree Regressor was tested to capture nonlinear relationships between property characteristics and `ClosePrice`.

The model used:

- `max_depth = 20`
- `min_samples_leaf = 5`
- `random_state = 42`

### Random Forest

A Random Forest Regressor was tested to improve upon a single decision tree by combining predictions from multiple trees.

The model used:

- `n_estimators = 50`
- `max_depth = 20`
- `min_samples_leaf = 5`
- `random_state = 42`

### XGBoost

An XGBoost Regressor was also tested to capture complex nonlinear relationships and interactions between property characteristics.

Hyperparameter tuning was performed using `GridSearchCV` with `TimeSeriesSplit`.

The parameters tested included:

- `n_estimators`
- `max_depth`
- `learning_rate`

The best parameters were:

- `n_estimators = 200`
- `max_depth = 5`
- `learning_rate = 0.1`

## Model Evaluation

The models were evaluated on the test set using:

- **R²** — measures the proportion of variation in `ClosePrice` explained by the model.
- **MAE (Mean Absolute Error)** — measures the average absolute difference between predicted and actual prices.
- **MAPE (Mean Absolute Percentage Error)** — measures the average percentage error.
- **MdAPE (Median Absolute Percentage Error)** — measures the median percentage error.

## Results

The model results on the test set were:

| Model | R² | MAE | MAPE | MdAPE |
|---|---:|---:|---:|---:|
| Linear Regression | -103.59 | $625,803.71 | 87.55% | 27.50% |
| Decision Tree | 0.75 | $254,873.07 | 26.22% | 10.95% |
| **Random Forest** | **0.77** | **$218,901.79** | **22.27%** | **9.13%** |
| XGBoost | 0.74 | $251,520.40 | 24.13% | 11.69% |

## Best Model

The **Random Forest Regressor** was the best-performing model among the models tested.

It achieved:

- **R²: 0.766**
- **MAE: $218,901.79**
- **MAPE: 22.27%**
- **MdAPE: 9.13%**

The Random Forest explained approximately **76.6% of the variation in property ClosePrice** on the test set and had the lowest error across all three percentage-based and absolute-error metrics.

Therefore, the **Random Forest model was selected as the final model** for predicting residential property ClosePrice.


## Instructions to Re-run the Code

The project is organized into Jupyter notebooks that should be run in order.

### 1. Install the required packages

Make sure Python is installed and install the required packages:

```bash
pip install pandas numpy scikit-learn xgboost geopandas shapely matplotlib

### 2. 

