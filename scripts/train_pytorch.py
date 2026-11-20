from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(
    ROOT / "data/processed/metformin_monthly.csv",
    parse_dates=["YEAR_MONTH"],
)
df["YEAR_MONTH"] = pd.to_datetime(df["YEAR_MONTH"])
df = df.sort_values("YEAR_MONTH").reset_index(drop=True)

df["TIME_INDEX"] = np.arange(len(df))
df["WEEKDAYS"] = df["YEAR_MONTH"].apply(
    lambda date: np.busday_count(
        date.strftime("%Y-%m-%d"),
        (date + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d"),
    )
)

# Same inputs as the regression: time, weekdays, month indicators.
month_columns = []
for month in range(2, 13):
    column = f"MONTH_{month}"
    df[column] = (df["YEAR_MONTH"].dt.month == month).astype(float)
    month_columns.append(column)

features = ["TIME_INDEX", "WEEKDAYS"] + month_columns
results = []
fig, axes = plt.subplots(3, 1, figsize=(11, 10))

for ax, train_end in zip(axes, [24, 30, 36]):
    torch.manual_seed(42)

    train = df.iloc[:train_end]
    test = df.iloc[train_end:train_end + 7]

    # Fit scaling ONLY on training data to avoid leakage.
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train = torch.tensor(
        x_scaler.fit_transform(train[features]), dtype=torch.float32
    )
    y_train = torch.tensor(
        y_scaler.fit_transform(train[["TOTAL_ITEMS"]]),
        dtype=torch.float32,
    )
    X_test = torch.tensor(
        x_scaler.transform(test[features]), dtype=torch.float32
    )

    # A small ANN: 13 inputs → 8 hidden neurons → 1 prediction.
    model = nn.Sequential(
        nn.Linear(len(features), 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=0.01
    )
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(800):
        predictions = model(X_train)
        loss = loss_fn(predictions, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        scaled_predictions = model(X_test).numpy()

    forecasts = y_scaler.inverse_transform(scaled_predictions).ravel()
    actual = test["TOTAL_ITEMS"].to_numpy()
    mae = np.abs(actual - forecasts).mean()

    results.append({"train_end": train_end, "pytorch_mae": mae})

    ax.plot(test["YEAR_MONTH"], actual, "o-", label="Actual")
    ax.plot(test["YEAR_MONTH"], forecasts, "o--", label="PyTorch")
    ax.set_title(f"Trained on {train_end} months | MAE: {mae:,.0f}")
    ax.set_ylabel("Prescription items")
    ax.legend()
    ax.grid(alpha=0.3)

results = pd.DataFrame(results)
print(results.round(2).to_string(index=False))

average_mae = results["pytorch_mae"].mean()
regression_mae = 22940.74  # Your previous result on these same splits.

print(f"\nPyTorch average MAE:    {average_mae:,.0f}")
print(f"Regression average MAE: {regression_mae:,.0f}")
print(
    "Improvement over regression: "
    f"{100 * (regression_mae - average_mae) / regression_mae:.1f}%"
)

fig.tight_layout()
output = ROOT / "pytorch_backtest.png"
fig.savefig(output, dpi=150)
plt.close(fig)
print(f"\nChart saved to: {output}")