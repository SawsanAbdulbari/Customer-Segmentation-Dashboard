import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def perform_clustering(data):
    # Feature Engineering
    snapshot_date = data['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = data.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
        'Invoice': 'nunique',  # Frequency
        'TotalPrice': 'sum'  # Monetary
    }).reset_index()
    rfm.columns = ['Customer ID', 'Recency', 'Frequency', 'Monetary']
    
    # Standardize RFM metrics
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    
    # Apply KMeans
    cluster_count = st.sidebar.slider("Select Number of Clusters", 2, 10, 3)
    kmeans = KMeans(n_clusters=cluster_count, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    return rfm

def visualize_clusters(rfm):
    st.header("Cluster Visualizations")
    
    # Scatterplot for Recency vs Monetary
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='Cluster', palette='viridis', s=100, ax=ax)
    plt.title('Recency vs Monetary Value')
    plt.xlabel('Recency (Days)')
    plt.ylabel('Monetary Value')
    st.pyplot(fig)
    
    # Scatterplot for Frequency vs Monetary
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=rfm, x='Frequency', y='Monetary', hue='Cluster', palette='viridis', s=100, ax=ax)
    plt.title('Frequency vs Monetary Value')
    plt.xlabel('Frequency')
    plt.ylabel('Monetary Value')
    st.pyplot(fig) 