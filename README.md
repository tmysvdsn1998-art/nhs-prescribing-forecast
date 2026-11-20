# nhs-prescribing-forecast
Predicting NHS prescription demand using machine learning and interactive dashboards

## Project Overview
The aim of this project is to predict prescription demands for medicines in the UK for sufficient budget allocation in the future years. Our primary target should be the monthly number of prescription items, with expenditure analysis as a secondary feature. The initial geographical scope will be in England. 

## Data Source
For the data source, we're planning to use publicly available prescribing data from the NHS Business Services Authority (NHSBSA).

## Project Roadmap

This project follows an end-to-end data science workflow, from NHS prescribing data acquisition and exploratory analysis to machine learning, forecasting, and interactive dashboard deployment.

| Issue | Phase | Key Objectives | Status |
|:---|:---|:---|:---|
| NHS-001 | Project Planning | Define the research question, scope, target variable, and success criteria | Complete |
| NHS-002 | GitHub Setup | Establish repository structure, version control, and development workflow | Complete |
| NHS-003 | Python Environment | Configure dependencies, testing, linting, and GitHub Actions CI | Complete |
| NHS-004 | API Integration | Integrate NHSBSA prescribing API, implement pagination, and develop automated tests | In Progress |
| NHS-005 | Data Collection & EDA | Collect historical data, clean and aggregate prescribing volumes, and visualise trends | In Progress |
| NHS-006 | Baseline Forecasting | Develop naive, seasonal naive, and moving-average forecasting models | Planned |
| NHS-007 | Advanced Forecasting | Train and optimise SARIMA and machine learning models such as XGBoost | Planned |
| NHS-008 | Model Evaluation | Evaluate forecasting accuracy using time-series cross-validation and error metrics | Planned |
| NHS-009 | Dashboard Development | Develop an interactive Streamlit dashboard for historical trends and forecasts | Planned |
| NHS-010 | Deployment & Documentation | Deploy the dashboard, document methodology, and prepare the final portfolio | Planned |

## Technology Stack

| Category | Technologies |
|:---|:---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Data Source | NHSBSA Open Data Portal |
| Statistical Modelling (Planned) | statsmodels, SARIMA |
| Machine Learning (Planned) | scikit-learn, XGBoost |
| Visualisation (Planned) | Matplotlib, Plotly |
| Dashboard (Planned) | Streamlit |
| Testing & Quality | pytest, Ruff |
| Version Control & CI | Git, GitHub Actions |
