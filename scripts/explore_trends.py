import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet(
    "data/processed/metformin_monthly.parquet"
)
df["DAYS_IN_MONTH"] = df["YEAR_MONTH"].dt.days_in_month

df["ITEMS_PER_DAY"] = (
    df["TOTAL_ITEMS"] / df["DAYS_IN_MONTH"]
)

print(
    df[["YEAR_MONTH", "TOTAL_ITEMS", "ITEMS_PER_DAY"]].head()
)

import numpy as np

df["WORKING_DAYS"] = df["YEAR_MONTH"].apply(
    lambda date: np.busday_count(
        date.strftime("%Y-%m-%d"),
        (date + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d")
    )
)

df["ITEMS_PER_WORKING_DAY"] = (
    df["TOTAL_ITEMS"] / df["WORKING_DAYS"]
)

print(
    df[[
        "YEAR_MONTH",
        "TOTAL_ITEMS",
        "WORKING_DAYS",
        "ITEMS_PER_WORKING_DAY"
    ]].head()
)

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

columns = [
    "TOTAL_ITEMS",
    "ITEMS_PER_DAY",
    "ITEMS_PER_WORKING_DAY"
]

titles = [
    "Monthly Prescription Items",
    "Items per Calendar Day",
    "Items per Weekday"
]

for ax, column, title in zip(axes, columns, titles):
    ax.plot(df["YEAR_MONTH"], df[column])
    ax.set_title(title)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

columns = [
    "TOTAL_ITEMS",
    "ITEMS_PER_DAY",
    "ITEMS_PER_WORKING_DAY"
]

for column in columns:
    mean = df[column].mean()
    variance = df[column].var()
    std = df[column].std()
    cv = (std / mean) * 100

    print(f"\n{column}")
    print(f"Variance: {variance:.2f}")
    print(f"Standard deviation: {std:.2f}")
    print(f"CV: {cv:.2f}%")

    df["YEAR_MONTH"] = pd.to_datetime(df["YEAR_MONTH"])
df = df.sort_values("YEAR_MONTH").reset_index(drop=True)

# Insert missing months so a 12-row shift means exactly one year.
df = (
    df.set_index("YEAR_MONTH")
    .asfreq("MS")
    .rename_axis("YEAR_MONTH")
    .reset_index()
)

# Compare against the same month last year.
df["YOY_TOTAL_PCT"] = (
    df["TOTAL_ITEMS"].pct_change(12, fill_method=None) * 100
)
df["YOY_WEEKDAY_PCT"] = (
    df["ITEMS_PER_WORKING_DAY"].pct_change(12, fill_method=None) * 100
)

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    df["YEAR_MONTH"],
    df["YOY_TOTAL_PCT"],
    marker="o",
    label="Monthly items",
)
ax.plot(
    df["YEAR_MONTH"],
    df["YOY_WEEKDAY_PCT"],
    marker="o",
    label="Items per weekday",
)

ax.axhline(0, color="black", linewidth=1)
ax.set_title("Metformin prescribing: change from the same month last year")
ax.set_ylabel("Year-on-year change (%)")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Rank the largest absolute changes after weekday adjustment.
# These are investigation candidates, not confirmed anomalies.
comparison = df.dropna(subset=["YOY_WEEKDAY_PCT"]).copy()
comparison["ABS_CHANGE"] = comparison["YOY_WEEKDAY_PCT"].abs()

print("\nLargest year-on-year changes after weekday adjustment:")
print(
    comparison.nlargest(8, "ABS_CHANGE")[
        [
            "YEAR_MONTH",
            "TOTAL_ITEMS",
            "WORKING_DAYS",
            "YOY_TOTAL_PCT",
            "YOY_WEEKDAY_PCT",
        ]
    ].to_string(index=False, float_format=lambda x: f"{x:.2f}")
)

months = pd.to_datetime([
    "2024-08-01", "2025-08-01",
    "2025-05-01", "2026-05-01",
])

print("\nInvestigating August and May:")
print(
    df.loc[
        df["YEAR_MONTH"].isin(months),
        [
            "YEAR_MONTH",
            "TOTAL_ITEMS",
            "WORKING_DAYS",
            "ITEMS_PER_WORKING_DAY",
            "YOY_TOTAL_PCT",
            "YOY_WEEKDAY_PCT",
        ],
    ].to_string(index=False, float_format=lambda x: f"{x:.2f}")
)

print("\nLowest year-on-year changes per weekday:")
print(
    df.dropna(subset=["YOY_WEEKDAY_PCT"])
    .nsmallest(5, "YOY_WEEKDAY_PCT")[
        [
            "YEAR_MONTH",
            "TOTAL_ITEMS",
            "YOY_TOTAL_PCT",
            "YOY_WEEKDAY_PCT",
        ]
    ]
    .to_string(index=False, float_format=lambda x: f"{x:.2f}")
)