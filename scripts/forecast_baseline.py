import pandas as pd

df = pd.read_parquet(
    "data/processed/metformin_monthly.parquet"
)

df = df.sort_values("YEAR_MONTH").reset_index(drop=True)

train = df.iloc[:-7].copy()
test = df.iloc[-7:].copy()

print("Training:", train["YEAR_MONTH"].min(), "to", train["YEAR_MONTH"].max())
print("Testing:", test["YEAR_MONTH"].min(), "to", test["YEAR_MONTH"].max())
print("Training rows:", len(train))
print("Testing rows:", len(test))

test["NAIVE_FORECAST"] = train["TOTAL_ITEMS"].iloc[-1]
print(test[["YEAR_MONTH", "TOTAL_ITEMS", "NAIVE_FORECAST"]])

test['ABS_ERROR'] = (test["TOTAL_ITEMS"] - test["NAIVE_FORECAST"]).abs().mean()
print("Mean Absolute Error (MAE):", test['ABS_ERROR'].iloc[0])

test['SEASONAL_FORECAST'] = train["TOTAL_ITEMS"].iloc[-12:-5].values
print(test[["YEAR_MONTH", "TOTAL_ITEMS", "SEASONAL_FORECAST"]])

test['ABS_ERROR'] = (test["TOTAL_ITEMS"] - test["SEASONAL_FORECAST"]).abs().mean()
print("Mean Absolute Error (MAE):", test['ABS_ERROR'].iloc[0])

test['SEASONAL_ERROR'] = (test["TOTAL_ITEMS"] - test["SEASONAL_FORECAST"])
print("Seasonal Errors:", test[["YEAR_MONTH", "TOTAL_ITEMS", "SEASONAL_FORECAST", "SEASONAL_ERROR"]])
Mean_Seasonal_Error = test['SEASONAL_ERROR'].mean()
print("Mean Seasonal Error:", Mean_Seasonal_Error)

sum_train_2024 = train[train["YEAR_MONTH"].dt.year == 2024]["TOTAL_ITEMS"].sum()
print("Total items in 2024:", sum_train_2024)
sum_train_2025 = train[train["YEAR_MONTH"].dt.year == 2025]["TOTAL_ITEMS"].sum()
print("Total items in 2025:", sum_train_2025) 

percentage_change = ((sum_train_2025 - sum_train_2024) / sum_train_2024) * 100
print("Percentage change from 2024 to 2025:", percentage_change, "%")

test['TREND_FORECAST'] = (test["SEASONAL_FORECAST"]) * (1 + percentage_change / 100)
print(test[["YEAR_MONTH", "SEASONAL_FORECAST", "TREND_FORECAST"]])
print('Mean Absolute Error (MAE):', (test["TOTAL_ITEMS"] - test["TREND_FORECAST"]).abs().mean())

MAPE = ((test["TOTAL_ITEMS"] - test["TREND_FORECAST"]).abs() / test["TOTAL_ITEMS"])
print("Mean Absolute Percentage Error (MAPE):", MAPE.mean() * 100, "%")


