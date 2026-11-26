import math
import os
import statistics
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

load_dotenv()

API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
if API_KEY:
    print("API key loaded successfully. API-Schlüssel erfolgreich geladen.")
else:
    raise ValueError(
        "API key is not set or missing. Please check your .env file. "
        "API-Schlüssel ist nicht gesetzt. Bitte überprüfen Sie Ihre .env-Datei."
    )

URL = "https://www.alphavantage.co/query"


def get_stock_data(symbol):
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": "compact",
        "apikey": API_KEY,
    }
    try:
        response = requests.get(URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as error:
        print(f"Error fetching data for {symbol}: {error}")
        print(f"Fehler beim Abrufen der Daten für {symbol}: {error}")
        return None

    if "Time Series (Daily)" not in data:
        print("API Error:", data)
        return None
    return data["Time Series (Daily)"]


ticker = input(
    "Enter the stock ticker symbol (e.g., AAPL for Apple): "
    "Geben Sie das Börsenticker-Symbol ein (z.B. AAPL für Apple): "
).strip().upper()

stock_data = get_stock_data(ticker)
if not stock_data:
    raise SystemExit("Exiting due to missing stock data.")

# Sort dates descending (newest first) and take the last 52 trading days
sorted_dates_desc = sorted(stock_data.keys(), reverse=True)[:52]
num_days = len(sorted_dates_desc)

print(f"Historical Stock Data for {ticker}: Aktienkursdaten für {ticker}:")
print("-" * 70)
for date in sorted_dates_desc:
    values = stock_data[date]
    print(
        f"Date: {date}, Open: {values['1. open']}, High: {values['2. high']}, "
        f"Low: {values['3. low']}, Close: {values['4. close']}, Volume: {values['5. volume']}"
    )

print("-" * 70)
print(f"Total number of days retrieved: {num_days}")
print(f"Here is the data for the last {num_days} days: Hier sind die Daten für die letzten {num_days} Tage:")

closing_prices_desc = [float(stock_data[d]["4. close"]) for d in sorted_dates_desc]
trading_volumes_desc = [int(stock_data[d]["5. volume"]) for d in sorted_dates_desc]

average_closing_price = statistics.mean(closing_prices_desc)
median_closing_price = statistics.median(closing_prices_desc)
max_closing_price = max(closing_prices_desc)
min_closing_price = min(closing_prices_desc)
std_dev_closing_price = statistics.stdev(closing_prices_desc)

average_volume = statistics.mean(trading_volumes_desc)
median_volume = statistics.median(trading_volumes_desc)
max_volume = max(trading_volumes_desc)
min_volume = min(trading_volumes_desc)
std_dev_volume = statistics.stdev(trading_volumes_desc)

print(f"Average Closing Price for the last {num_days} days: {average_closing_price:.2f}")
print(f"Median Closing Price for the last {num_days} days: {median_closing_price:.2f}")
print(f"Average Trading Volume for the last {num_days} days: {average_volume:.2f}")
print(f"Median Trading Volume for the last {num_days} days: {median_volume:.2f}")
print(f"Max Closing Price for the last {num_days} days: {max_closing_price:.2f}")
print(f"Min Closing Price for the last {num_days} days: {min_closing_price:.2f}")
print(f"Max Trading Volume for the last {num_days} days: {max_volume}")
print(f"Min Trading Volume for the last {num_days} days: {min_volume}")
print(f"Standard Deviation of Closing Prices for the last {num_days} days: {std_dev_closing_price:.2f}")
print(f"Standard Deviation of Trading Volumes for the last {num_days} days: {std_dev_volume:.2f}")

# Chronological order (oldest first) for return & time-series calculations
closing_prices = closing_prices_desc[::-1]
daily_returns = [
    (closing_prices[i] - closing_prices[i - 1]) / closing_prices[i - 1]
    for i in range(1, len(closing_prices))
]
log_returns = [
    math.log(closing_prices[i] / closing_prices[i - 1])
    for i in range(1, len(closing_prices))
]
daily_std_dev = statistics.stdev(daily_returns)
annual_historical_volatility = daily_std_dev * math.sqrt(252)

print(f"Standard Deviation of Daily Returns for the last {num_days} days: {daily_std_dev:.4f}")
print(f"Daily Log Returns for the last {num_days} days: {log_returns}")
print(f"Annual Historical Volatility for the last {num_days} days: {annual_historical_volatility:.4f}")

# Linear Regression Model (uses full available history in stock_data for more training rows)
all_dates_asc = sorted(stock_data.keys())
all_closes_asc = [float(stock_data[d]["4. close"]) for d in all_dates_asc]

df = pd.DataFrame({"Close": all_closes_asc}, index=pd.to_datetime(all_dates_asc))
print(f"Total Observations: {len(df)}")
print(f"First Date: {df.index.min().date()}")
print(f"Last Date: {df.index.max().date()}")
print(f"Missing Values: {df.isnull().sum().sum()}")
print(f"Duplicate Dates: {df.index.duplicated().sum()}")

if not df.index.is_monotonic_increasing:
    raise ValueError("Dates are not in chronological order.")
if df.index.has_duplicates:
    raise ValueError("Detected Duplicate Dates")
if df.isnull().values.any():
    raise ValueError("Missing values detected.")

df["Daily_Return"] = df["Close"].pct_change()
df["Return_5D"] = df["Close"].pct_change(5)
df["SMA_5"] = df["Close"] / df["Close"].rolling(5).mean()
df["SMA_10"] = df["Close"] / df["Close"].rolling(10).mean()
df["Volatility_5D"] = df["Daily_Return"].rolling(5).std() * math.sqrt(252)
df["Future_5D_Return"] = (df["Close"].shift(-5) / df["Close"]) - 1
df["Future_5D_Direction"] = (df["Future_5D_Return"] > 0).astype(int)

features = [
    "Daily_Return",
    "Return_5D",
    "SMA_5",
    "SMA_10",
    "Volatility_5D",
]

# Most recent row with valid features (for predicting the next 5 days from today)
valid_features_df = df[features].dropna()
latest_X = valid_features_df.iloc[[-1]]

# Rows with both valid features and known 5-day forward target (for training/evaluation)
train_df = df[features + ["Future_5D_Return"]].dropna()
X = train_df[features]
y = train_df["Future_5D_Return"]

split = int(len(train_df) * 0.8)
# Purge 5 rows before `split` so target windows crossing the boundary don't leak into training
purge_end = split - 5
if purge_end < 10 or len(X.iloc[split:]) == 0:
    raise ValueError("Not enough observations for reliable model validation.")

X_train = X.iloc[:purge_end]
y_train = y.iloc[:purge_end]

X_test = X.iloc[split:]
y_test = y.iloc[split:]

model = LinearRegression()
model.fit(X_train, y_train)
test_predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, test_predictions)
rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
direction_accuracy = np.mean(np.sign(test_predictions) == np.sign(y_test)) * 100

zero_predictions = np.zeros(len(y_test))
mean_predictions = np.full(len(y_test), y_train.mean())
zero_mae = mean_absolute_error(y_test, zero_predictions)
mean_mae = mean_absolute_error(y_test, mean_predictions)

zero_rmse = np.sqrt(mean_squared_error(y_test, zero_predictions))
mean_rmse = np.sqrt(mean_squared_error(y_test, mean_predictions))

print("-" * 45, "MODEL VALIDATION RESULTS", "-" * 45)
print(f"Training observations: {len(X_train)}")
print(f"Testing observations: {len(X_test)}")
print(f"MAE: {mae:.4%}")
print(f"RMSE: {rmse:.4%}")
print(f"Direction Accuracy: {direction_accuracy:.2f}%")

print(f"Zero Baseline MAE:  {zero_mae:.4%}")
print(f"Mean Baseline MAE: {mean_mae:.4%}")
print(f"Zero Baseline RMSE: {zero_rmse:.4%}")
print(f"Mean Baseline RMSE: {mean_rmse:.4%}")

final_model = LinearRegression()
final_model.fit(X, y)
next_prediction = float(final_model.predict(latest_X)[0])

print("-" * 45, "MODEL PREDICTION RESULTS", "-" * 45)
print(f"Predicted 5-day Return: {next_prediction:.4%}")
if next_prediction > 0:
    print("The model predicts a POSITIVE return for the next 5 days. Das Modell sagt eine POSITIVE Rendite für die nächsten 5 Tage voraus.")
elif next_prediction < 0:
    print("The model predicts a NEGATIVE return for the next 5 days. Das Modell sagt eine NEGATIVE Rendite für die nächsten 5 Tage voraus.")
else:
    print("The model predicts a NEUTRAL return for the next 5 days. Das Modell sagt eine NEUTRALE Rendite für die nächsten 5 Tage voraus.")

print("-" * 45, "MODEL COEFFICIENTS", "-" * 45)
for feature, coefficient in zip(features, final_model.coef_):
    print(f"{feature}: {coefficient:.6f}")
print(f"Intercept: {final_model.intercept_:.6f}")

# Model 2: Momentum Model
momentum_df = pd.DataFrame(
    {"Close": all_closes_asc},
    index=pd.to_datetime(all_dates_asc),
)
momentum_df["Return_5D"] = momentum_df["Close"].pct_change(5)
momentum_df["Return_10D"] = momentum_df["Close"].pct_change(10)

# Simple Moving Averages
momentum_df["SMA_5"] = momentum_df["Close"].rolling(5).mean()
momentum_df["SMA_10"] = momentum_df["Close"].rolling(10).mean()

# Compare short-term and long-term moving averages
momentum_df["SMA_Ratio"] = (momentum_df["SMA_5"] / momentum_df["SMA_10"]) - 1

# Remove incomplete observations
momentum_df = momentum_df.dropna()

latest_momentum = momentum_df.iloc[-1]

return_5d = latest_momentum["Return_5D"]
return_10d = latest_momentum["Return_10D"]
sma_ratio = latest_momentum["SMA_Ratio"]

momentum_score = (return_5d + return_10d + sma_ratio) / 3

print("-" * 45, "MOMENTUM MODEL RESULTS", "-" * 45)
print(f"5-Day Return: {return_5d:.4%}")
print(f"10-Day Return: {return_10d:.4%}")
print(f"SMA Ratio: {sma_ratio:.4%}")
print(f"Momentum Score: {momentum_score:.4%}")

if momentum_score > 0:
    print("Momentum Direction: POSITIVE")
elif momentum_score < 0:
    print("Momentum Direction: NEGATIVE")
else:
    print("Momentum Direction: NEUTRAL")

print("-" * 45)

# Logistic Regression
logistic_df = df[features + ["Future_5D_Direction"]].dropna()
X_logistic = logistic_df[features]
y_logistic = logistic_df["Future_5D_Direction"]

logistic_split = int(len(logistic_df) * 0.8)
logistic_purge_end = logistic_split - 5

if logistic_purge_end < 10 or logistic_split >= len(logistic_df):
    raise ValueError("Not enough observations for Logistic Regression validation.")

X_logistic_train = X_logistic.iloc[:logistic_purge_end]
y_logistic_train = y_logistic.iloc[:logistic_purge_end]

X_logistic_test = X_logistic.iloc[logistic_split:]
y_logistic_test = y_logistic.iloc[logistic_split:]

logistic_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
)

logistic_model.fit(X_logistic_train, y_logistic_train)
logistic_predictions = logistic_model.predict(X_logistic_test)
logistic_probabilities = logistic_model.predict_proba(X_logistic_test)[:, 1]

accuracy = accuracy_score(y_logistic_test, logistic_predictions)
precision = precision_score(y_logistic_test, logistic_predictions, zero_division=0)
recall = recall_score(y_logistic_test, logistic_predictions, zero_division=0)
f1 = f1_score(y_logistic_test, logistic_predictions, zero_division=0)

print("-" * 45, "LOGISTIC REGRESSION RESULTS", "-" * 45)

print(f"Training observations: {len(X_logistic_train)}")
print(f"Testing observations: {len(X_logistic_test)}")
print(f"Accuracy: {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1 Score: {f1:.4f}")

if y_logistic_test.nunique() == 2:
    auc = roc_auc_score(y_logistic_test, logistic_probabilities)
    print(f"ROC-AUC: {auc:.4f}")
else:
    print("ROC-AUC unavailable: test set contains only one class")

print("Confusion Matrix:")
print(confusion_matrix(y_logistic_test, logistic_predictions))

majority_class = y_logistic_train.mode().iloc[0]

baseline_predictions = np.full(len(y_logistic_test), majority_class)
baseline_accuracy = accuracy_score(y_logistic_test, baseline_predictions)

print(f"Majority Baseline Accuracy: {baseline_accuracy:.2%}")

final_logistic_model = LogisticRegression(max_iter=1000, class_weight="balanced")
final_logistic_model.fit(X_logistic, y_logistic)

latest_probability = final_logistic_model.predict_proba(latest_X[features])[0, 1]

print("-" * 45, " LATEST CLASSIFICATION ", "-" * 45)
print(f"Probability of Positive 5-Day Return: {latest_probability:.2%}")
print(f"Probability of Non-Positive 5-Day Return: {1 - latest_probability:.2%}")

print("-" * 45)

# Linear Regression predicts the 5-day return (expected direction and magnitude).
# Momentum Model shows the momentum score (recent price trend).
# Logistic Regression shows the probability of a positive 5-day return.
#
# BUY if Linear Regression > 0 AND Momentum Score > 0 AND Logistic probability >= 0.60
# SELL if Linear Regression < 0 AND Momentum Score < 0 AND Logistic probability <= 0.40
# HOLD otherwise.

if next_prediction > 0 and momentum_score > 0 and latest_probability >= 0.6:
    decision = "BUY"
elif next_prediction < 0 and momentum_score < 0 and latest_probability <= 0.4:
    decision = "SELL"
else:
    decision = "HOLD"

print("-" * 45, "FINAL DECISION", "-" * 45)
print(f"Based on statistics and after a lot of calculations. We recommend that you: {decision}")
print("Thank you for using Zahlenwerk3.0 Quantitative Research software!")
print("-" * 45)


#Instructions on how to interpret and use the data from Zahlenwerk3.0 Quantiative Research Software.
"""
What return does the Linear Regression Model estimate over the next 5 trading days?
If postive number = estimated positive return.
If Negative number = estimated negative return.
Zero = estimated no change

Momentum model helps us determine what direction has the stock's recent trend been moving
If postive score = positive recent momentum.
If negative score = negative recent momentum.
Near zero score = weaker or mixed momentum.

Logistic Regression helps us determine what probability does the model assign to a positive 5 day return
If above 50% = positive outcome is more likely accordiong to the model.
Below 50% = non-positive outcome is more likely according to the model.

There are different measurements. In particular, the momentum score is not a prediction of the future percentage return.

Calculations:
Daily Return =  How much the closing price changed compared with the previous day.
5 Day Return = Price change over 5 trading days
10 Day Return = Price change over 10 trading days
SMA 5 = Average closing price over 5 trading days
SMA 10 = Average clsing price over 10 trading days
Volatility = How much daily returns have varied.
Future 5 Day Return = Actual price change over the follwing next 5 trading days.
SMA = Simple Moving Average
MAE = Average size of prediction errors
RMSE = Error measure that penalizes larger mistakes more heavily
Directional Accuracy = Precentage of predictions with the correct direction.
Accuracy = Percentage of classifications that were correct
Precision = How often positive predictions were actually positive
Recall = How many acutal positive outcomes were identified.
F1 = Balance between precision and recall.
ROC-AUC = How well the model ranks positive outcomes above negative ones.


How to interpret the findings?
+ return = estimated positive price movement
- return = estimated negative price movement
+ momentum = recent upward trend
- momentum = recent downward trend
Higher probability = stronger estimated chance of a positive outcome.
Lower MAE or RMSE = smaller regression errors.
Higher accuracy/F1 Score =n better classigication performance, depending on class balance. 
HOLD decision = insufficient agreement under the rules and calculations.
"""
