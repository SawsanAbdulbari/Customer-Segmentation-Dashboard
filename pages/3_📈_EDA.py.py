import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

# Title of the dashboard
st.title("Cluster Visualizations")

data = pd.read_csv(r'data/online_retail_II.csv', encoding='latin1')

# Clean the data
cleaned_data = data.dropna(subset=['Customer ID'])      
cleaned_data['InvoiceDate'] = pd.to_datetime(cleaned_data['InvoiceDate'])
cleaned_data = cleaned_data[(cleaned_data['Quantity'] > 0) & (cleaned_data['Price'] > 0)]
cleaned_data['TotalPrice'] = cleaned_data['Quantity'] * cleaned_data['Price']

# Filter options
st.sidebar.header("Choose your filter: ")
# Create for Country
country = st.sidebar.multiselect("Pick Country", cleaned_data["Country"].unique())
if not country:
    filtered_data = cleaned_data.copy()
else:
    filtered_data = cleaned_data[cleaned_data["Country"].isin(country)]

# Feature Engineering
snapshot_date = filtered_data['InvoiceDate'].max() + dt.timedelta(days=1)
rfm = filtered_data.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
        'Invoice': 'nunique',  # Frequency
        'TotalPrice': 'sum'  # Monetary
    }).reset_index()
rfm.columns = ['Customer ID', 'Recency', 'Frequency', 'Monetary']

    # Standardize RFM metrics
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# Select number of clusters
st.sidebar.header("Clustering Options")
cluster_count = st.sidebar.slider("Select Number of Clusters", 2, 10, 3)

# Apply KMeans
kmeans = KMeans(n_clusters=cluster_count, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)



# Visualize the clusters
# Scatterplot for Recency vs Monetary
fig = px.scatter(rfm, x='Recency', y='Monetary', animation_frame='Cluster', color='Cluster', size='Frequency', hover_name='Customer ID')
st.plotly_chart(fig, use_container_width=True)

# Scatterplot for Frequency vs Monetary
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=rfm, x='Frequency', y='Monetary', hue='Cluster', palette='viridis', s=100, ax=ax)
plt.title('Frequency vs Monetary Value')
plt.xlabel('Frequency')
plt.ylabel('Monetary Value')
st.pyplot(fig)

    # Cluster profile bar chart
cluster_profiles = rfm.groupby('Cluster').agg({
        'Recency': 'mean',  # Recency
        'Frequency': 'mean',  # Frequency
        'Monetary': 'mean'  # Monetary
    }).reset_index()

cluster_profiles_melted = cluster_profiles.melt(id_vars='Cluster', var_name='Metric', value_name='Value')

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=cluster_profiles_melted, x='Metric', y='Value', hue='Cluster', palette='viridis', ax=ax)
plt.title('Cluster Profiles (Mean RFM Metrics)')
plt.xlabel('RFM Metric')
plt.ylabel('Mean Value')
st.pyplot(fig)

    # Calculate summary statistics
num_customers = filtered_data['Customer ID'].nunique()
num_invoices = filtered_data['Invoice'].nunique()
total_quantity = filtered_data['Quantity'].sum()
total_price = filtered_data['Price'].sum()

# Hierarchical view of Loan
# st.subheader("Hierarchical view of Loan")

#     # Filter out rows with zero TotalPrice
# filtered_df = cleaned_data[cleaned_data['TotalPrice'] > 0]

#     # Create the treemap figure
# fig3 = px.treemap(filtered_df,  
#                     path=["Country", "Description"],  # Adjusted to available columns
#                     values="TotalPrice",        
#                     hover_data=["TotalPrice"],
#                     color="TotalPrice",  # Adjust the color based on TotalPrice
#                     color_continuous_scale='Viridis',  # Choose a color scale
#                     )

#     # Customize the layout
# fig3.update_layout(
#     width=800,
#     height=650,
#     margin=dict(l=0, r=0, b=0, t=30),  # Adjust margin to make space for subheader
#     )

#     # Add more interactivity
# fig3.update_traces(
#     hoverinfo="label+value+percent parent",
#     textinfo="label+value",
#     )

#     # Set the title and axis labels
# fig3.update_layout(
#         title="Loan Funding by Country and Description",
#         xaxis_title="Funded Amount",
#         yaxis_title="Category",
#     )

# st.plotly_chart(fig3, use_container_width=True)

    # Convert the "InvoiceDate" column to datetime data type
cleaned_data["InvoiceDate"] = pd.to_datetime(cleaned_data["InvoiceDate"])

# Now you can use the .dt accessor to extract the month name
cleaned_data["month_name"] = cleaned_data["InvoiceDate"].dt.month_name()

# st.subheader(":point_right: Monthly Activity Loan Summary")
# with st.expander("Summary_Table"):
#     df_sample = cleaned_data[0:5][["Country", "Description", "TotalPrice"]]
#     fig = ff.create_table(df_sample, colorscale="Cividis")
#     st.plotly_chart(fig, use_container_width=True)

# st.markdown("Month wise Activity Table")
# sub_category_Year = pd.pivot_table(data=cleaned_data, 
#                                     values="TotalPrice", 
#                                     index=["Description"],
#                                     columns="month_name")
# st.write(sub_category_Year.style.background_gradient(cmap="Blues"))


# Show the cluster data
st.write("RFM Segmentation with Clusters:")
st.dataframe(rfm)