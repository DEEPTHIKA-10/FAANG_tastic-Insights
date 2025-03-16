# FAANG_tastic-Insights
FAANG Stock Closing Price Predictor
## Overview
This project is a **Streamlit-based web application** that predicts the closing price of FAANG stocks
(Facebook, Apple, Amazon, Netflix, and Google) based on given market inputs.
The model uses **machine learning** techniques, including data preprocessing and MLflow
integration for model tracking.
## Features
- **Interactive UI** using Streamlit.
- **Stock Price Prediction** based on input data.
- **MLflow Tracking** for model version control.
- **Pickle-based Encoding** for categorical data.
## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/faang-stock-predictor.git
cd faang-stock-predictor
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the application:
```bash
streamlit run app.py
```
## Usage
1. Select a **FAANG stock** (Amazon, Apple, Facebook, Google, or Netflix).
2. Enter stock details (**Open, High, Low, Volume**).
3. Click **Predict the Closing Price**.
4. View the **Predicted Closing Price** on the screen.
## Files Description
- `app.py` - Streamlit app for stock price prediction.
- `FAANG.csv` - Dataset used for training the model.
- `standard_scaler.pkl`, `one_hot_encoder.pkl` - Pre-trained encoders for data transformation.
- `models:/Random Forest Regressor/1` - MLflow-tracked stock prediction model.
- 
