import streamlit as st
import pandas as pd

def load_and_clean_data():
    data = pd.read_csv(r'data\online_retail_II.csv', encoding='latin1')
    cleaned_data = data.dropna(subset=['Customer ID'])
    cleaned_data['InvoiceDate'] = pd.to_datetime(cleaned_data['InvoiceDate'])
    cleaned_data = cleaned_data[(cleaned_data['Quantity'] > 0) & (cleaned_data['Price'] > 0)]
    cleaned_data['TotalPrice'] = cleaned_data['Quantity'] * cleaned_data['Price']
    return cleaned_data
