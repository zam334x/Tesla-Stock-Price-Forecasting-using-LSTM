# Tesla Stock Price Forecasting using LSTM

## Overview
This project involves time series forecasting of Tesla’s stock price using Long Short-Term Memory (LSTM) neural networks. The model is designed to predict the "Close" price of Tesla's stock based on historical stock data, using deep learning techniques to identify trends and patterns.
- **Please see the "Tesla_Stock_Forecasting.ipynb" file for full details and results.** 

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Data Preprocessing](#data-preprocessing)
- [Modeling](#modeling)
- [Performance Evaluation](#performance-evaluation)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Installation](#installation)
- [Usage](#usage)
- [Contributors](#contributors)

## Dataset
The dataset contains historical data of Tesla Inc.'s stock price (TSLA), covering daily levels. It includes the following columns:
- **Date**: Date of the trading day
- **Open**: Opening price of the stock
- **High**: Highest price of the stock during the day
- **Low**: Lowest price of the stock during the day
- **Close**: Closing price of the stock
- **Adj Close**: Adjusted closing price (adjusted for dividends, stock splits, etc.)
- **Volume**: Number of shares traded

The target variable for this project is the **Close** price.

## Project Structure
The project is structured as follows:

## Data Preprocessing
1. **Exploratory Data Analysis (EDA)**:
   - Visualized trends in Open, Close, and Volume columns to understand data patterns.
   
2. **Normalization**:
   - Used MinMaxScaler to scale the "Close" price between 0 and 1 for better performance during model training.

3. **Data Preparation**:
   - Split data into training (75%) and testing (25%) sets.
   - Created sequences of 60 time steps (using a sliding window technique) as input for the model.

## Modeling
### LSTM Model
- **Architecture**:
  - The model consists of two LSTM layers followed by Dense layers.
  - ReLU activation is used for Dense layers, and mean squared error (MSE) is used as the loss function.
  - Early stopping is implemented to prevent overfitting.

### Hyperparameters
- **Epochs**: 100
- **Batch size**: 32
- **Learning rate**: Adaptive based on the Adam optimizer.

## Performance Evaluation
The following metrics were used to evaluate model performance:
- **Root Mean Squared Error (RMSE)**
- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **R-squared (R²) Score**
- **Mean Absolute Percentage Error (MAPE)**

The LSTM model showed promising results with a low RMSE and a high R² score, indicating accurate prediction of Tesla’s stock price trends.

## Results
The LSTM model achieved the following results:
- **RMSE**: 23.82
- **MSE**: 1891.92
- **MAE**: 27.35
- **R-squared (R²)**: 0.98
- **MAPE**: 5.1%

The predictions aligned well with actual stock prices, demonstrating the model's effectiveness in forecasting short-term price movements.

## Future Improvements
- **Hyperparameter Tuning**: Use tools like GridSearchCV or Keras Tuner to fine-tune model parameters.
- **Additional Features**: Incorporate moving averages, technical indicators, or macroeconomic factors to enhance accuracy.
- **Advanced Models**: Experiment with models like Bidirectional LSTM, CNN-LSTM, or Transformer-based models for better performance.
- **Cross-Validation**: Implement k-fold cross-validation for more robust performance evaluation.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/tesla-stock-forecasting.git
   cd tesla-stock-forecasting
