import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def visualize_clusters(rfm):
    st.header("Cluster Visualizations")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='Cluster', palette='viridis', s=100, ax=ax)
    plt.title('Recency vs Monetary Value')
    plt.xlabel('Recency (Days)')
    plt.ylabel('Monetary Value')
    st.pyplot(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=rfm, x='Frequency', y='Monetary', hue='Cluster', palette='viridis', s=100, ax=ax)
    plt.title('Frequency vs Monetary Value')
    plt.xlabel('Frequency')
    plt.ylabel('Monetary Value')
    st.pyplot(fig) 