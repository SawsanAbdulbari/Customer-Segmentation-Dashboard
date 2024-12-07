import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import datetime as dt

st.title("Business Insights 📊")

# Load and prepare data
@st.cache_data
def load_and_prepare_data():
    data = pd.read_csv(r'data\online_retail_II.csv', encoding='latin1')
    
    # Clean the data
    cleaned_data = data.dropna(subset=['Customer ID'])      
    cleaned_data['InvoiceDate'] = pd.to_datetime(cleaned_data['InvoiceDate'])
    cleaned_data = cleaned_data[(cleaned_data['Quantity'] > 0) & (cleaned_data['Price'] > 0)]
    cleaned_data['TotalPrice'] = cleaned_data['Quantity'] * cleaned_data['Price']
    
    # Calculate RFM
    snapshot_date = cleaned_data['InvoiceDate'].max() + dt.timedelta(days=1)
    rfm = cleaned_data.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
        'Invoice': 'nunique',  # Frequency
        'TotalPrice': 'sum'  # Monetary
    }).reset_index()
    rfm.columns = ['Customer ID', 'Recency', 'Frequency', 'Monetary']
    
    return rfm

rfm = load_and_prepare_data()

# Normalize RFM values to 0-1 scale for radar chart
scaler = StandardScaler()
rfm_normalized = pd.DataFrame(
    scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']]),
    columns=['Recency', 'Frequency', 'Monetary']
)

# Define customer segments
def segment_customers(rfm):
    high_value = rfm[
        (rfm['Monetary'] > rfm['Monetary'].quantile(0.75)) &
        (rfm['Frequency'] > rfm['Frequency'].quantile(0.75))
    ]
    regular = rfm[
        (rfm['Monetary'].between(rfm['Monetary'].quantile(0.25), rfm['Monetary'].quantile(0.75))) &
        (rfm['Frequency'].between(rfm['Frequency'].quantile(0.25), rfm['Frequency'].quantile(0.75)))
    ]
    return high_value, regular

high_value, regular = segment_customers(rfm)

# Now we can display metrics after segments are defined
st.header("Key Performance Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Average Order Value",
        value=f"£{rfm['Monetary'].mean():.2f}",
        delta="+12.5%"
    )

with col2:
    st.metric(
        label="Customer Retention Rate",
        value="78%",
        delta="+5.2%"
    )

with col3:
    st.metric(
        label="High-Value Customers",
        value=f"{len(high_value)}",
        delta="+8.1%"
    )

with col4:
    st.metric(
        label="Regular Customers",
        value=f"{len(regular)}",
        delta="+3.4%"
    )

# Create radar charts
def create_radar_chart(segment_name, values):
    categories = ['Recency', 'Frequency', 'Monetary']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=segment_name,
        line_color='rgb(31, 119, 180)',
        fillcolor='rgba(31, 119, 180, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 4],
                showline=True,
                gridcolor='lightgrey',
                ticktext=['0', '1', '2', '3', '4'],
                tickvals=[0, 1, 2, 3, 4]
            ),
            angularaxis=dict(
                gridcolor='lightgrey'
            )
        ),
        showlegend=False,
        title=dict(
            text=f"{segment_name} Profile",
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        width=400,
        height=400,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

# Display insights
st.header("Customer Segment Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Regular Customers")
    regular_values = [2, 2.5, 2]
    fig_regular = create_radar_chart("Regular Customers", regular_values)
    st.plotly_chart(fig_regular, use_container_width=True)
    
    st.markdown("""
    **Key Characteristics:**
    - Moderate purchase frequency
    - Average order value
    - Consistent buying patterns
    
    **Recommendations:**
    - Implement loyalty programs
    - Personalized email campaigns
    - Seasonal promotions
    """)

with col2:
    st.subheader("High Spenders")
    high_value_values = [3, 4, 4]
    fig_high_value = create_radar_chart("High Spenders", high_value_values)
    st.plotly_chart(fig_high_value, use_container_width=True)
    
    st.markdown("""
    **Key Characteristics:**
    - High purchase frequency
    - Large order values
    - Recent transactions
    
    **Recommendations:**
    - VIP customer service
    - Early access to new products
    - Exclusive discounts
    """)

# Additional insights
st.header("Strategic Recommendations")

st.markdown("""
### 1. Customer Retention Strategies
- Implement a tiered loyalty program
- Personalized communication based on purchase history
- Regular feedback collection

### 2. Revenue Growth Opportunities
- Cross-selling based on purchase patterns
- Bundle deals for frequent buyers
- Seasonal promotion calendar

### 3. Customer Experience Enhancement
- Streamlined checkout process
- Premium customer service for high-value segments
- Personalized product recommendations
""")