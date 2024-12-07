import streamlit as st
import warnings

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(page_title='Customer Segmentation Dashboard', 
                   page_icon=":bar_chart:",
                   layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    body {
        font-family: 'Roboto', sans-serif;
    }
    
    .block-container {
        padding-top: 1rem;
    }
    
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        transition: background-color 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #45a049;
    }
    
    .stSidebar {
        background-color: #f0f2f6;
    }
    
    .stSidebar .stButton>button {
        background-color: #007BFF;
        color: white;
        transition: background-color 0.3s ease;
    }
    
    .stSidebar .stButton>button:hover {
        background-color: #0056b3;
    }
    
    .stSidebar .stImage {
        margin-bottom: 1rem;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-weight: 500;
    }
    
    /* Add icon styles */
    .icon {
        margin-right: 8px;
    }
    
    </style>
    """, unsafe_allow_html=True)

# Title and introduction with icon
st.title(":bar_chart: Customer Segmentation Dashboard")
st.markdown("""
**Welcome to the Customer Segmentation Dashboard** :wave:

*Your gateway to in-depth exploration and analysis of customer data.*

**Overview:**

This interactive dashboard empowers you to delve into customer segmentation data. Our goal is to display the data research and analysis by offering insightful information on customer behavior.
""", unsafe_allow_html=True)

# Sidebar for additional information
st.sidebar.title('Dashboard Information :information_source:')

# Update the image path to the correct location
image = r"images/output.png"
st.sidebar.image(image, caption='Image', use_column_width=True)

st.sidebar.write("This dashboard provides insights into customer segmentation using RFM analysis.")
st.sidebar.write("")

# Signature
st.sidebar.write("")
st.sidebar.markdown("Made with :green_heart: by [Sawsan Abdulbari](https://www.linkedin.com/in/sawsanabdulbari/)")

# Subheader and introduction
st.subheader("Exploration and Analysis :mag:")

st.markdown('''
**Features:**

1. **Insights:** Explore crucial statistics and metrics related to customer segmentation, including RFM analysis and clustering visualizations.
2. **Navigation:** The left-hand sidebar menu makes it easy to navigate through various sections, including "Overview" and "Data Exploration & Analysis."
3. **Filtering:** The left-hand sidebar menu has a wide variety of data filtering options for desired specific insights.

**Data Exporting:** 

The unfiltered and filtered data can be downloaded in .csv format.

**Data Sources:**

We source data meticulously from our internal datasets, ensuring data integrity and reliability.

**Conclusion and Summary:**
In this dashboard, we have presented key insights from the customer segmentation data analysis. We observed trends in customer behavior and analyzed the distribution of customer segments.

**Acknowledgment:**

We extend our gratitude to the team and community for their valuable contributions.
''', unsafe_allow_html=True)

# Increase font size for the entire introduction text
st.markdown('<div style="font-size: 24px;">---</div>', unsafe_allow_html=True)

