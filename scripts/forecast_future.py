from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(
    ROOT / "data/processed/metformin_monthly.csv",
    parse_dates=["YEAR_MONTH"],
).sort_values("YEAR_MONTH").reset_index(drop=True)

start = df["YEAR_MONTH"].min()


def make_features(dates):
    X = pd.DataFrame(index=dates.index)

    X["TIME_INDEX"] = (
        (dates.dt.year - start.year) * 12
        + dates.dt.month - start.month
    )

    X["WEEKDAYS"] = dates.apply(
        lambda date: np.busday_count(
            date.strftime("%Y-%m-%d"),
            (date + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d"),
        )
    )

    for month in range(2, 13):
        X[f"MONTH_{month}"] = (dates.dt.month == month).astype(int)

    return X


model = LinearRegression()
model.fit(make_features(df["YEAR_MONTH"]), df["TOTAL_ITEMS"])

future = pd.DataFrame({
    "YEAR_MONTH": pd.date_range(
        start=df["YEAR_MONTH"].max() + pd.offsets.MonthBegin(1),
        periods=6,
        freq="MS",
    )
})

future["FORECAST_ITEMS"] = model.predict(
    make_features(future["YEAR_MONTH"])
).round().astype(int)

output_dir = ROOT / "outputs"
output_dir.mkdir(exist_ok=True)

future.to_csv(output_dir / "metformin_forecast.csv", index=False)

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    df["YEAR_MONTH"], df["TOTAL_ITEMS"],
    label="Historical", color="steelblue",
)

# Connect the forecast to the last observed month.
forecast_dates = pd.concat(
    [df["YEAR_MONTH"].tail(1), future["YEAR_MONTH"]],
    ignore_index=True,
)
forecast_values = np.concatenate([
    df["TOTAL_ITEMS"].tail(1).to_numpy(),
    future["FORECAST_ITEMS"].to_numpy(),
])

ax.plot(
    forecast_dates, forecast_values,
    "o--", label="Six-month forecast", color="darkorange",
)

ax.set_title("Metformin prescribing: six-month regression forecast")
ax.set_ylabel("Prescription items")
ax.ticklabel_format(axis="y", style="plain")
ax.legend()
ax.grid(alpha=0.3)

fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(output_dir / "metformin_forecast.png", dpi=150)
plt.close(fig)

print("\nSix-month forecast:")
print(future.to_string(index=False))
print(f"\nSaved chart and CSV to {output_dir}")