from nhs_forecast.api import fetch_prescribing_data
import pandas as pd


records = fetch_prescribing_data(
    resource_id="EPD_SNOMED_202607",
    bnf_code="0601022B0",
    limit=100,
    offset=0,
)

df = pd.DataFrame(records)

print(df.head())
print("\nShape:", df.shape)
print("\nColumns:", df.columns.tolist())

print("\nITEMS data type:")
print(df["ITEMS"].dtype)

print("\nMissing ITEMS values:")
print(df["ITEMS"].isna().sum())

print("\nSummary statistics:")
print(df["ITEMS"].describe())

monthly_items = (
    df.groupby("YEAR_MONTH")["ITEMS"]
    .sum()
    .reset_index()
)

print(monthly_items)