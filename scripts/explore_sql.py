import json
import urllib.parse
import urllib.request
import urllib.error

base_url = "https://opendata.nhsbsa.net/api/3/action/datastore_search_sql"


def fetch_monthly_total(resource_id, bnf_code):
    
    year_month = int(resource_id.rsplit("_", 1)[-1])

    if year_month < 202503:
        code_column = "BNF_CHEMICAL_SUBSTANCE"
    else:
        code_column = "BNF_CHEMICAL_SUBSTANCE_CODE"
    sql = f"""
            SELECT
                YEAR_MONTH,
                SUM(ITEMS) AS TOTAL_ITEMS
            FROM `{resource_id}`
            WHERE {code_column} = '{bnf_code}'
            GROUP BY YEAR_MONTH
            """

    params = {
        "resource_id": resource_id,
        "sql": sql,
    }

    url = base_url + "?" + urllib.parse.urlencode(params)

    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        print(f"\nFailed resource: {resource_id}")
        print(f"HTTP status: {error.code}")
        print("NHSBSA response:")
        print(error.read().decode("utf-8"))
        raise

    return data["result"]["result"]["records"][0]

def inspect_schema(resource_id):
    sql = f"""
    SELECT *
    FROM `{resource_id}`
    LIMIT 1
    """

    params = {
        "resource_id": resource_id,
        "sql": sql,
    }

    url = base_url + "?" + urllib.parse.urlencode(params)

    with urllib.request.urlopen(url, timeout=60) as response:
        data = json.loads(response.read().decode("utf-8"))

    records = data["result"]["result"]["records"]

    return list(records[0].keys())

import pandas as pd

months = pd.period_range(
    start="2025-02",
    end="2025-12",
    freq="M"
)

import pandas as pd

months = pd.period_range(
    start="2023-01",
    end="2026-07",
    freq="M"
)

results = []

for month in months:
    resource_id = f"EPD_SNOMED_{month.strftime('%Y%m')}"
    print(f"Fetching {month}...")

    result = fetch_monthly_total(resource_id, "0601022B0")
    results.append(result)

df = pd.DataFrame(results)

# Standardise both date formats
df["YEAR_MONTH"] = (
    df["YEAR_MONTH"]
    .astype(str)
    .str.replace("-", "", regex=False)
)

df["YEAR_MONTH"] = pd.to_datetime(
    df["YEAR_MONTH"],
    format="%Y%m"
)

df = df.sort_values("YEAR_MONTH").reset_index(drop=True)

print(df.head())
print(df.dtypes)

print(df)
print("Number of months:", len(df))
print("Missing values:")
print(df.isna().sum())

print("Duplicate months:", df["YEAR_MONTH"].duplicated().sum())
print("Number of months:", len(df))

from pathlib import Path

output_path = Path("data/processed/metformin_monthly.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_path, index=False)
df.to_parquet(
    "data/processed/metformin_monthly.parquet",
    index=False
)

print(f"Saved to {output_path}")