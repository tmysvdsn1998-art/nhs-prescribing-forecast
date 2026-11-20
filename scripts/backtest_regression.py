import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
df = pd.read_csv(
    "data/processed/metformin_monthly.csv",
    parse_dates=["YEAR_MONTH"],
)

df = df.sort_values("YEAR_MONTH").reset_index(drop=True)
df["TIME_INDEX"] = range(len(df))
#print(df[["YEAR_MONTH", "TIME_INDEX"]].tail())
df["MONTH"] = df["YEAR_MONTH"].dt.month

df["WEEKDAYS"] = df["YEAR_MONTH"].apply(
    lambda date: np.busday_count(
        date.strftime("%Y-%m-%d"),
        (date + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d")
    )
)


df = pd.get_dummies(
    df,
    columns=["MONTH"],
    drop_first=True,
    dtype=int
)

features = ["TIME_INDEX"] + [f"MONTH_{i}" for i in range(2, 13)]
feature_sets = {
    "Original": features,
    "With weekdays": features + ["WEEKDAYS"],
    "Random forest": features + ["WEEKDAYS"],
}

backtest_results = []

for train_end in [24, 30, 36]:
    train = df.iloc[:train_end].copy()
    test = df.iloc[train_end:train_end + 7].copy()

    for name, model_features in feature_sets.items():
        if name == "Random forest":
            model = RandomForestRegressor(
                n_estimators=300,
                max_depth=4,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )
        else:
            model = LinearRegression()        
        model.fit(train[model_features], train["TOTAL_ITEMS"])

        predictions = model.predict(test[model_features])
        mae = (test["TOTAL_ITEMS"] - predictions).abs().mean()
        if name == "With weekdays":
            errors = test[["YEAR_MONTH", "TOTAL_ITEMS"]].copy()
            errors["FORECAST"] = predictions
            errors["ERROR"] = errors["TOTAL_ITEMS"] - predictions
            errors["ABS_ERROR"] = errors["ERROR"].abs()

            print(f"\nLargest errors: training on {train_end} months")
            print(
                errors.nlargest(3, "ABS_ERROR").to_string(
                    index=False,
                    float_format=lambda x: f"{x:,.0f}",
                )
            )        
        backtest_results.append({
            "train_end": train_end,
            "model": name,
            "mae": mae,
        })

results_df = pd.DataFrame(backtest_results)

comparison = results_df.pivot(
    index="train_end",
    columns="model",
    values="mae",
)

comparison["Improvement_pct"] = (
    (comparison["Original"] - comparison["With weekdays"])
    / comparison["Original"]
    * 100
)

print("\nMAE comparison:")
print(comparison.round(2))

print("\nAverage MAE across splits:")
print(
    results_df.groupby("model")["mae"]
    .mean()
    .sort_values()
    .round(2)
)

fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(
    errors["YEAR_MONTH"], errors["TOTAL_ITEMS"],
    marker="o", label="Actual"
)
ax.plot(
    errors["YEAR_MONTH"], errors["FORECAST"],
    marker="o", linestyle="--", label="Forecast"
)

ax.set_title(f"Weekday model — trained on {train_end} months")
ax.set_ylabel("Prescription items")
ax.legend()
ax.grid(alpha=0.3)
fig.autofmt_xdate()
fig.tight_layout()

fig.savefig(f"backtest_{train_end}.png", dpi=150)
plt.close(fig)