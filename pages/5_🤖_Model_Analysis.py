import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import datetime as dt
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

st.title("Model Analysis 🤖")

# Load and prepare data
@st.cache_data(ttl=3600, show_spinner="Loading data...")
def load_and_prepare_data():
    try:
        # Use more efficient data reading with specific dtypes
        dtypes = {
            'Customer ID': str,
            'Invoice': str,
            'Price': float,
            'Quantity': int
        }
        parse_dates = ['InvoiceDate']
        
        data = pd.read_csv(
            r'data\online_retail_II.csv', 
            encoding='latin1',
            dtype=dtypes,
            parse_dates=parse_dates,
            usecols=['Customer ID', 'Invoice', 'InvoiceDate', 'Price', 'Quantity']  # Only load needed columns
        )
        
        # Vectorized operations instead of apply
        cleaned_data = data.dropna(subset=['Customer ID'])
        cleaned_data = cleaned_data[(cleaned_data['Quantity'] > 0) & (cleaned_data['Price'] > 0)]
        cleaned_data['TotalPrice'] = cleaned_data['Quantity'] * cleaned_data['Price']
        
        # More efficient RFM calculation
        snapshot_date = cleaned_data['InvoiceDate'].max() + pd.Timedelta(days=1)
        rfm = cleaned_data.groupby('Customer ID').agg({
            'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
            'Invoice': 'nunique',
            'TotalPrice': 'sum'
        }).reset_index()
        rfm.columns = ['Customer ID', 'Recency', 'Frequency', 'Monetary']
        
        return rfm
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load data
rfm = load_and_prepare_data()

# Move Model Parameters to sidebar
st.sidebar.header("Model Parameters")

n_clusters = st.sidebar.slider(
    "Number of Clusters", 
    min_value=2, 
    max_value=8, 
    value=4,
    help="Choose the number of customer segments to create"
)

random_state = st.sidebar.number_input(
    "Random State", 
    min_value=0, 
    max_value=100, 
    value=42,
    help="Set random seed for reproducible results"
)

# Add additional sidebar controls for better user experience
st.sidebar.markdown("---")
show_advanced = st.sidebar.checkbox("Show Advanced Metrics", value=False)
max_display_points = st.sidebar.slider(
    "Max Points in 3D Plot", 
    min_value=1000, 
    max_value=10000, 
    value=5000,
    help="Limit the number of points shown in 3D visualization for better performance"
)

# Prepare data for clustering
scaler = StandardScaler()
features_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# Train KMeans model
@st.cache_resource(show_spinner="Training model...")
def train_kmeans(n_clusters, random_state, features_scaled):
    try:
        kmeans = KMeans(
            n_clusters=n_clusters, 
            random_state=random_state,
            n_init=10,  # Explicitly set n_init
            max_iter=300,  # Limit maximum iterations
            tol=1e-4  # Set convergence tolerance
        )
        kmeans.fit(features_scaled)
        return kmeans
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None

kmeans = train_kmeans(n_clusters, random_state, features_scaled)

# Add cluster labels to the dataframe
rfm['Cluster'] = kmeans.labels_

@st.cache_data
def get_cluster_characteristics(rfm_data):
    cluster_means = rfm_data.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    overall_means = rfm_data[['Recency', 'Frequency', 'Monetary']].mean()
    return cluster_means, overall_means

def assign_cluster_names_vectorized(rfm_data):
    cluster_means, overall_means = get_cluster_characteristics(rfm_data)
    
    conditions = [
        (cluster_means['Frequency'] > overall_means['Frequency']) & 
        (cluster_means['Monetary'] > overall_means['Monetary']) & 
        (cluster_means['Recency'] < overall_means['Recency']),
        
        (cluster_means['Frequency'] > overall_means['Frequency']) & 
        (cluster_means['Monetary'] > overall_means['Monetary']),
        
        (cluster_means['Frequency'] > overall_means['Frequency']),
        
        (cluster_means['Recency'] < overall_means['Recency']) & 
        (cluster_means['Monetary'] > overall_means['Monetary']),
        
        (cluster_means['Recency'] > overall_means['Recency'] * 1.5),
        
        (cluster_means['Monetary'] < overall_means['Monetary'] * 0.5)
    ]
    
    choices = [
        'Champions',
        'Loyal Customers',
        'Regular Customers',
        'Recent High Spenders',
        'Lost Customers',
        'Low-Value Customers'
    ]
    
    default = 'Average Customers'
    
    cluster_names = pd.Series(default, index=cluster_means.index)
    for condition, choice in zip(conditions, choices):
        cluster_names[condition] = choice
    
    return cluster_names[rfm_data['Cluster']].values

# Add cluster names to the dataframe
rfm['Segment_Name'] = assign_cluster_names_vectorized(rfm)

# Model Evaluation Metrics
st.header("Model Evaluation")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="Inertia Score",
        value=f"{kmeans.inertia_:.2f}",
        help="Sum of squared distances of samples to their closest cluster center"
    )

with col2:
    st.metric(
        label="Number of Iterations",
        value=kmeans.n_iter_,
        help="Number of iterations run to converge"
    )

# Cluster Visualization
st.header("Cluster Visualization")

# 3D Scatter Plot
def create_3d_scatter(rfm_data, max_points):
    if len(rfm_data) > max_points:
        sampled_data = rfm_data.groupby('Segment_Name', group_keys=False).apply(
            lambda x: x.sample(min(len(x), max_points // len(rfm_data['Segment_Name'].unique())))
        )
    else:
        sampled_data = rfm_data
    
    fig = px.scatter_3d(
        sampled_data,
        x='Recency',
        y='Frequency',
        z='Monetary',
        color='Segment_Name',
        title=f'Customer Segments in 3D Space (Showing {len(sampled_data):,} of {len(rfm_data):,} points)',
        labels={'Segment_Name': 'Customer Segment'},
        width=800,
        height=600
    )
    
    return fig

scatter_fig = create_3d_scatter(rfm, max_display_points)
st.plotly_chart(scatter_fig)

# Show advanced metrics if enabled
if show_advanced:
    st.header("Advanced Model Metrics")
    
    advanced_cols = st.columns(3)
    
    with advanced_cols[0]:
        st.metric(
            label="Silhouette Score",
            value=f"{silhouette_score(features_scaled, kmeans.labels_):.3f}",
            help="Measure of how similar an object is to its own cluster compared to other clusters"
        )
    
    with advanced_cols[1]:
        st.metric(
            label="Calinski-Harabasz Score",
            value=f"{calinski_harabasz_score(features_scaled, kmeans.labels_):.0f}",
            help="Ratio of between-cluster dispersion and within-cluster dispersion"
        )
    
    with advanced_cols[2]:
        st.metric(
            label="Davies-Bouldin Score",
            value=f"{davies_bouldin_score(features_scaled, kmeans.labels_):.3f}",
            help="Average similarity measure of each cluster with its most similar cluster"
        )

# Cluster Characteristics
st.header("Cluster Characteristics")

cluster_means = rfm.groupby('Segment_Name')[['Recency', 'Frequency', 'Monetary']].mean()

# Create a heatmap of cluster characteristics
fig_heatmap = px.imshow(
    cluster_means,
    labels=dict(x="Metrics", y="Segment", color="Value"),
    title="Segment Characteristics Heatmap"
)
st.plotly_chart(fig_heatmap)

# Cluster Analysis
st.header("Cluster Analysis")

for segment in rfm['Segment_Name'].unique():
    with st.expander(f"{segment} Analysis"):
        cluster_data = rfm[rfm['Segment_Name'] == segment]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Average Recency (days)",
                value=f"{cluster_data['Recency'].mean():.0f}"
            )
        
        with col2:
            st.metric(
                label="Average Frequency",
                value=f"{cluster_data['Frequency'].mean():.1f}"
            )
        
        with col3:
            st.metric(
                label="Average Monetary Value",
                value=f"£{cluster_data['Monetary'].mean():.2f}"
            )
        
        st.markdown(f"""
        **Segment Size:** {len(cluster_data)} customers ({(len(cluster_data)/len(rfm)*100):.1f}% of total)
        
        **Characteristics:**
        - {'High' if cluster_data['Recency'].mean() < rfm['Recency'].mean() else 'Low'} recency (more recent = lower value)
        - {'High' if cluster_data['Frequency'].mean() > rfm['Frequency'].mean() else 'Low'} frequency
        - {'High' if cluster_data['Monetary'].mean() > rfm['Monetary'].mean() else 'Low'} monetary value
        """)

# Prediction Interface
st.header("Customer Segment Predictor")
st.markdown("Enter customer metrics to predict their segment:")

col1, col2, col3 = st.columns(3)

with col1:
    input_recency = st.number_input("Recency (days)", min_value=0, value=30)
    
with col2:
    input_frequency = st.number_input("Frequency", min_value=1, value=5)
    
with col3:
    input_monetary = st.number_input("Monetary Value", min_value=0.0, value=100.0)

if st.button("Predict Segment"):
    try:
        # Scale the input
        input_scaled = scaler.transform([[input_recency, input_frequency, input_monetary]])
        
        # Predict
        prediction = kmeans.predict(input_scaled)[0]
        
        # Get segment name by comparing with cluster characteristics
        cluster_means, overall_means = get_cluster_characteristics(rfm)
        
        # Create a temporary series with the prediction metrics
        temp_metrics = pd.Series({
            'Recency': input_recency,
            'Frequency': input_frequency,
            'Monetary': input_monetary
        })
        
        # Compare with overall means to determine segment
        conditions = [
            (input_frequency > overall_means['Frequency']) and 
            (input_monetary > overall_means['Monetary']) and 
            (input_recency < overall_means['Recency']),
            
            (input_frequency > overall_means['Frequency']) and 
            (input_monetary > overall_means['Monetary']),
            
            (input_frequency > overall_means['Frequency']),
            
            (input_recency < overall_means['Recency']) and 
            (input_monetary > overall_means['Monetary']),
            
            (input_recency > overall_means['Recency'] * 1.5),
            
            (input_monetary < overall_means['Monetary'] * 0.5)
        ]
        
        choices = [
            'Champions',
            'Loyal Customers',
            'Regular Customers',
            'Recent High Spenders',
            'Lost Customers',
            'Low-Value Customers'
        ]
        
        # Find the first matching condition
        segment_name = next((choice for condition, choice in zip(conditions, choices) if condition), 'Average Customers')
        
        st.success(f"Predicted Customer Segment: {segment_name}")
        
        # Show characteristics of predicted segment
        st.markdown("### Predicted Segment Characteristics:")
        segment_stats = rfm[rfm['Segment_Name'] == segment_name].describe()
        st.dataframe(segment_stats[['Recency', 'Frequency', 'Monetary']])
        
        # Show comparison with segment averages
        st.markdown("### Comparison with Segment Averages:")
        segment_avg = rfm[rfm['Segment_Name'] == segment_name][['Recency', 'Frequency', 'Monetary']].mean()
        
        comparison_df = pd.DataFrame({
            'Your Customer': [input_recency, input_frequency, input_monetary],
            'Segment Average': segment_avg.values
        }, index=['Recency', 'Frequency', 'Monetary'])
        
        st.dataframe(comparison_df)
        
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")

# Download Section
st.header("Download Results")

# Prepare download data
download_data = rfm.copy()
download_data = download_data.drop('Cluster', axis=1)  # Remove numeric cluster labels

# Convert to CSV
csv = download_data.to_csv(index=False)

st.download_button(
    label="Download Segmentation Results",
    data=csv,
    file_name="customer_segments.csv",
    mime="text/csv"
) 