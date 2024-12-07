import streamlit as st

def setup_download_buttons(data):
    st.sidebar.header("Download Options")
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Data as CSV", data=csv, file_name="data.csv", mime="text/csv") 