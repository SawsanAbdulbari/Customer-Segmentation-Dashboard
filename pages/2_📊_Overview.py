import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from util.data_processing import load_and_clean_data

# Load and clean data
data = load_and_clean_data()

def display_overview(data):
    # Calculate metrics
    num_customers = data['Customer ID'].nunique()
    num_invoices = data['Invoice'].nunique()
    total_quantity = data['Quantity'].sum()
    total_price = data['TotalPrice'].sum()
    
    # Display metrics
    st.title("Data Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Number of Customers", num_customers)
    col2.metric("Number of Invoices", num_invoices)
    col3.metric("Total Quantity", total_quantity)
    col4.metric("Total Price", f"${total_price:,.2f}")
    
    # Filter options
    st.sidebar.header("Choose your filter: ")
    # Create for Country
    country = st.sidebar.multiselect("Pick Country", data["Country"].unique())
    if not country:
        filtered_data = data.copy()
    else:
        filtered_data = data[data["Country"].isin(country)]
    # Sidebar layout for date range
    st.sidebar.header("Date Range Filters:")
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2009-12-01 07:45:00'))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime('2011-12-09 12:50:00'))
    
    # Date range validation
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    if start_date > end_date:
        st.error("Start date should be before the end date.")
        return
    
   # Sample data if necessary
    max_points = 1000  # Define a maximum number of points for visualization
    if len(filtered_data) > max_points:
        sampled_data = filtered_data.sample(max_points)
    else:
        sampled_data = filtered_data

    # Create a layout with two columns for the bar and pie charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Create a bar plot using Plotly Express
        st.subheader("Total Price by Country")
        country_price = sampled_data.groupby('Country')['TotalPrice'].sum().reset_index()
        fig = px.bar(country_price, x='Country', y='TotalPrice',
                     labels={'Country': 'Country', 'TotalPrice': 'Total Price'},
                     text=['${:,.2f}'.format(x) for x in country_price['TotalPrice']],
                     template="seaborn", color='Country')
        st.plotly_chart(fig, use_container_width=True, responsive=True)

    with chart_col2:
        # Create a pie chart using Plotly Express
        st.subheader("Country Distribution of Transactions")
        country_counts = sampled_data['Country'].value_counts().reset_index()
        country_counts.columns = ['Country', 'Count']
        fig = px.pie(country_counts, names='Country', values='Count',
                     title='Country Distribution of Transactions',
                     template="seaborn")
        fig.update_traces(textinfo='percent+label', pull=[0.1] * len(country_counts))
        st.plotly_chart(fig, use_container_width=True, responsive=True)
    
    # Define a function for data view and download
    def view_and_download_data(data, download_filename, download_label):
        st.write(data.style.background_gradient(cmap="Blues"))
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(download_label, 
                        data=csv, 
                        file_name=download_filename, 
                        mime="text/csv", 
                        help=f'Click here to download the {download_label} as a CSV file')
    
    # Define the layout of columns
    cl1, cl2 = st.columns(2)
    
    # Sector Data
    with cl1:
        with st.expander("Sector_ViewData"):
            sector_data = sampled_data.groupby('Country')['TotalPrice'].sum().reset_index()
            view_and_download_data(sector_data, "Sector.csv", "Download Sector Data")
    
    # Country Data
    with cl2:
        with st.expander("Country_ViewData"):
            country_data = sampled_data.groupby('Country')['TotalPrice'].sum().reset_index()
            view_and_download_data(country_data, "Country.csv", "Download Country Data")
    
    # Time series data aggregation
    time_series_data = sampled_data.groupby(sampled_data['InvoiceDate'].dt.strftime("%Y-%b"))['TotalPrice'].sum().reset_index()
    
    # Create time series chart
    st.subheader("Time Series Analysis of Total Price")
    st.markdown("This dashboard allows you to analyze sales data over a specified date range.")
    fig2 = px.line(time_series_data, x='InvoiceDate', y="TotalPrice", markers=True, labels={"TotalPrice": "Amount"})
    fig2.update_xaxes(type='category')
    fig2.update_layout(height=500, width=1000, template="plotly_white")
    
    # Add a trendline
    trendline = go.Scatter(x=time_series_data['InvoiceDate'], y=time_series_data['TotalPrice'],
                        mode='lines', line=dict(color='red'), name='Trendline')
    fig2.add_trace(trendline)
    
    # Add an annotation
    fig2.add_annotation(x="2017-Jul", y=3000, 
                        text="Significant Decrease", 
                        showarrow=True, arrowhead=1,
                        arrowsize=1.5, arrowwidth=2)
    
    st.plotly_chart(fig2, use_container_width=True, responsive=True)
    
    # Data preprocessing
    filtered_data['InvoiceDate'] = pd.to_datetime(filtered_data['InvoiceDate'])
    filtered_data = filtered_data[(filtered_data['InvoiceDate'] >= start_date) & (filtered_data['InvoiceDate'] <= end_date)]
    
    # Sort dataset by highest and lowest price
    sorted_by_highest_price = filtered_data.sort_values(by='Price', ascending=False).head(5)
    sorted_by_lowest_price = filtered_data.sort_values(by='Price', ascending=True).head(5)
    
    st.write("Our dataset, sorted by highest price:")
    st.dataframe(sorted_by_highest_price)
    
    st.write("Our dataset, sorted by lowest price:")
    st.dataframe(sorted_by_lowest_price)
    
    
# Ensure the function is called
if __name__ == "__main__":
    try:
        display_overview(data)
    except Exception as e:
        st.error(f"An error occurred: {e}") 