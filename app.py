import streamlit as st
import pandas as pd
import numpy as np
import pickle
import mlflow
import mlflow.pyfunc
import matplotlib.pyplot as plt

std_scaler = pickle.load(open(r'C:\Users\kumar\Desktop\FAANG_PROJECT\standard_scaler.pkl','rb'))
one_hot_encoder = pickle.load(open(r'C:\Users\kumar\Desktop\FAANG_PROJECT\one_hot_encoder.pkl','rb'))
debt_to_equity_label_encoder = pickle.load(open(r'C:\Users\kumar\Desktop\FAANG_PROJECT\debt_to_equity_label_encoder.pkl','rb'))
eps_label_encoder = pickle.load(open(r'C:\Users\kumar\Desktop\FAANG_PROJECT\eps_label_encoder.pkl','rb'))
market_cap_label_encoder = pickle.load(open(r'C:\Users\kumar\Desktop\FAANG_PROJECT\market_cap_label_encoder.pkl','rb'))
pe_ratio_label_encoder = pickle.load(open(r'C:\Users\kumar\Desktop\FAANG_PROJECT\pe_ratio_label_encoder.pkl','rb'))
price_to_book_ratio_label_encoder = pickle.load(open(r'C:\Users\kumar\Desktop\FAANG_PROJECT\price_to_book_ratio_label_encoder.pkl','rb'))

mlflow.set_tracking_uri("http://127.0.0.1:5000")
model_name = "Random Forest Regressor"
model_version = "1"
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.pyfunc.load_model(model_uri)

st.title("FAANG - Stock Closing Price Predictor")

st.markdown("""
This streamlit application predicts the closing price of FAANG stocks based on the provided input data.
""")

st.header("Input Stock Details")

# Input fields for user to enter stock data
company_name = st.selectbox("Company", ["Amazon (AMZN)", "Apple (AAPL)", "Facebook (META)", "Google (GOOGL)", "Netflix (NFLX)"])
open = st.number_input("Open Price")
high = st.number_input("High Price")
low = st.number_input("Low Price")
volume = st.number_input("Volume")

if company_name == "Amazon (AMZN)":
    company = 'Amazon'
    market_cap = '2000000000000.0'
    pe_ratio = '45.496414'
    eps = '4.18'
    debt_to_equity = '66.756'
    price_to_book_ratio = '8.4372225'
elif company_name == "Apple (AAPL)":
    company = 'Apple'
    market_cap = '2845000000000.0'
    pe_ratio = '35.789955'
    eps = '6.57'
    debt_to_equity = '139.49249999999998'
    price_to_book_ratio = '23.000308749999995'
elif company_name == "Facebook (META)":
    company = 'Facebook'
    market_cap = '1470000000000.0'
    pe_ratio = '29.612986'
    eps = '19.56'
    debt_to_equity = '24.235'
    price_to_book_ratio = '9.359326'
elif company_name == "Google (GOOGL)":
    company = 'Google'
    market_cap = '2020000000000.0'
    pe_ratio = '23.492826'
    eps = '6.97'
    debt_to_equity = '9.549'
    price_to_book_ratio = '6.7086606'
elif company_name == "Netflix (NFLX)":
    company = 'Netflix'
    market_cap = '645000000000.0'
    pe_ratio = '42.8245'
    eps = '17.67'
    debt_to_equity = '70.338'
    price_to_book_ratio = '14.262457'

# Label encoding
market_cap_encoded = int(market_cap_label_encoder.transform([market_cap])[0])
pe_ratio_encoded = int(pe_ratio_label_encoder.transform([pe_ratio])[0])
eps_encoded = int(eps_label_encoder.transform([eps])[0])
debt_to_equity_encoded = int(debt_to_equity_label_encoder.transform([debt_to_equity])[0])
price_to_book_ratio_label_encoded = int(price_to_book_ratio_label_encoder.transform([price_to_book_ratio])[0])

# one hot encoding
company_encoded = one_hot_encoder.transform([[company]]).toarray()
Amazon = int(company_encoded[0][0])
Apple = int(company_encoded[0][1])
Facebook = int(company_encoded[0][2])
Google = int(company_encoded[0][3])
Netflix = int(company_encoded[0][4])

features = [[open, high, low, volume, market_cap_encoded, pe_ratio_encoded, eps_encoded, debt_to_equity_encoded, price_to_book_ratio_label_encoded, Apple, Facebook, Google, Amazon, Netflix]]
scaled_features = std_scaler.transform(features)

if st.button("Predict the Closing Price"):
    predicted_closing_price = model.predict(scaled_features)[0]
    st.header("Predicted Closing Price")
    st.subheader(f"${predicted_closing_price:.4f}")