# Zahlenwerk 3.0 — Quantitative Research Software

## Overview

Zahlenwerk 3.0 is a Python-based quantitative research application that analyzes historical stock market data using statistical methods and machine learning. It retrieves financial data through the Alpha Vantage API and combines three analytical models to examine potential five-day stock price movements.

## Features

The software combines Linear Regression to estimate future five-day returns, a Momentum Model to analyze recent price trends, and Logistic Regression to estimate the probability of a positive five-day return.

It also calculates historical volatility, daily returns, logarithmic returns, moving averages, and descriptive statistics.

The three models are evaluated using statistical performance metrics and combined into a rule-based BUY, SELL, or HOLD classification.

## Technologies

Developed using Python, Zahlenwerk 3.0 utilizes NumPy, Pandas, Scikit-learn, Requests, Statistics, and python-dotenv. Historical financial data is retrieved through the Alpha Vantage API.

## Installation

Clone the repository and install the required dependencies:

```bash
pip install numpy pandas requests scikit-learn python-dotenv
```

Create a `.env` file in the project directory and add your Alpha Vantage API key:

```env
ALPHA_VANTAGE_API_KEY=your_api_key_here
```

Run the application:

```bash
python main.py
```

## Model Validation

Zahlenwerk uses chronological training and testing datasets, with a five-observation purge to reduce overlapping target-window leakage across the validation boundary.

Linear Regression is evaluated using MAE, RMSE, and directional accuracy. Logistic Regression is evaluated using accuracy, precision, recall, F1 score, and ROC-AUC.

## Limitations

Zahlenwerk 3.0 is an experimental research application. Its models use relatively limited historical data and cannot account for every factor influencing financial markets. Predictions and probabilities are estimates, not guarantees. Transaction costs and market liquidity are not currently incorporated.

## Future Development

Potential improvements include walk-forward validation, historical backtesting, Sharpe Ratio calculations, maximum drawdown analysis, probability calibration, and more advanced statistical models.

## Disclaimer

Zahlenwerk 3.0 is intended exclusively for educational, experimental, and quantitative research purposes. It does not provide financial advice or guarantee investment returns.

---

Zahlenwerk 3.0 — Mathematics. Statistics. Data. Research.
