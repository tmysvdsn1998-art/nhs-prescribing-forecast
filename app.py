from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="NHS Prescribing Forecast",
    page_icon="📊",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
history_path = ROOT / "data/processed/metformin_monthly.csv"
forecast_path = ROOT / "outputs/metformin_forecast.csv"

if not history_path.exists() or not forecast_path.exists():
    st.error("Historical data or forecast CSV is missing. Run the data and forecast scripts first.")
    st.stop()

history = pd.read_csv(history_path, parse_dates=["YEAR_MONTH"])
forecast = pd.read_csv(forecast_path, parse_dates=["YEAR_MONTH"])

history = history.sort_values("YEAR_MONTH")
forecast = forecast.sort_values("YEAR_MONTH")

# Recorded development-backtest results, not recalculated by the app.
scores = pd.DataFrame({
    "Model": [
        "Regression + weekdays",
        "PyTorch ANN",
        "Random Forest + weekdays",
        "Original regression",
    ],
    "MAE": [22940.74, 29101.00, 48744.35, 52090.39],
})

st.title("NHS Metformin Prescribing Forecast")
cutoff = history["YEAR_MONTH"].max()
st.caption(
    f"Based on data through {cutoff:%B %Y} · "
    "Forecast model: regression with time trend, month and weekday count"
)

horizon = st.sidebar.slider(
    "Forecast months to display",
    min_value=1,
    max_value=len(forecast),
    value=len(forecast),
)
visible_forecast = forecast.head(horizon)

latest = history.iloc[-1]
peak = visible_forecast.loc[
    visible_forecast["FORECAST_ITEMS"].idxmax()
]

col1, col2, col3 = st.columns(3)
col1.metric(
    f"Latest actual · {latest['YEAR_MONTH']:%b %Y}",
    f"{latest['TOTAL_ITEMS']:,.0f}",
)
col2.metric("Selected model backtest MAE", "22,941 items")
col3.metric(
    f"Peak forecast · {peak['YEAR_MONTH']:%b %Y}",
    f"{peak['FORECAST_ITEMS']:,.0f}",
)

trends_tab, models_tab, data_tab = st.tabs(
    ["Trends & forecast", "Model comparison", "Data & download"]
)

with trends_tab:
    st.subheader("Historical prescribing and forecast")

    actual_series = history.set_index("YEAR_MONTH")[
        "TOTAL_ITEMS"
    ].rename("Historical")

    forecast_series = visible_forecast.set_index("YEAR_MONTH")[
        "FORECAST_ITEMS"
    ].rename("Forecast")

    # Keep actual observations separate from predicted values.
    chart_data = pd.concat(
        [actual_series, forecast_series], axis=1
    )

    st.line_chart(
        chart_data,
        x_label="Month",
        y_label="Prescription items",
        color=["#2563eb", "#f97316"],
    )

    st.info(
        "These are point forecasts, not observed prescribing volumes. "
        "Prediction intervals have not yet been estimated."
    )

with models_tab:
    st.subheader("Development backtest performance")
    st.caption("Lower mean absolute error (MAE) is better.")

    st.bar_chart(scores.set_index("Model")["MAE"])
    st.dataframe(scores, hide_index=True)

    st.markdown(
        "**Weekday regression reduced average MAE by 56%** "
        "compared with the original regression."
    )
    st.caption(
        "Results use three overlapping seven-month test windows with "
        "24, 30 and 36 training months. These windows informed model "
        "selection; they are not an untouched final evaluation. "
        "Scores are recorded experiment results. The ANN result is rounded."
    )

with data_tab:
    st.subheader("Forecast values")
    st.dataframe(visible_forecast, hide_index=True)

    st.download_button(
        "Download displayed forecast",
        data=visible_forecast.to_csv(index=False),
        file_name="metformin_forecast.csv",
        mime="text/csv",
    )

    with st.expander("View historical data"):
        st.dataframe(history, hide_index=True)

st.caption(
    "Portfolio research project · Not an official NHS dashboard · "
    "Weekday counts exclude weekends but do not account for bank holidays."
)