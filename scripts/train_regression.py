import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression 

df = pd.read_parquet(
    "data/processed/metformin_monthly.parquet"
)

df = df.sort_values("YEAR_MONTH").reset_index(drop=True)
df["TIME_INDEX"] = range(len(df))
print(df[["YEAR_MONTH", "TIME_INDEX"]].tail())
df["MONTH"] = df["YEAR_MONTH"].dt.month

df = pd.get_dummies(
    df,
    columns=["MONTH"],
    drop_first=True,
    dtype=int
)

features = ["TIME_INDEX"] + [f"MONTH_{i}" for i in range(2, 13)]

print(features)


train = df.iloc[:-7].copy()
test = df.iloc[-7:].copy()

X_train = train[features]
y_train = train["TOTAL_ITEMS"]  

X_test = test[features]
y_test = test["TOTAL_ITEMS"]

print(X_train.shape, X_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)

print(model.intercept_)
print(model.coef_)

test['LINEAR_FORECAST'] = model.predict(X_test)
print(test[["YEAR_MONTH", "TOTAL_ITEMS", "LINEAR_FORECAST"]])
print('Mean Absolute Error (MAE):', (test["TOTAL_ITEMS"] - test["LINEAR_FORECAST"]).abs().mean())
