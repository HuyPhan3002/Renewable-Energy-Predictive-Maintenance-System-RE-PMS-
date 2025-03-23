import streamlit as st
import pandas as pd

# Set page title
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

st.title("🔧 Predictive Maintenance Dashboard")

# Load predictions
data_path = r"D:\class\CS480\predictions.csv"
try:
    df = pd.read_csv(data_path)

    st.subheader("⚠️ Machines at Risk of Failure")
    st.dataframe(df, use_container_width=True)

    st.write("✅ *Take immediate action to prevent downtime.*")
except FileNotFoundError:
    st.warning("No failure predictions found. Run the ML script first.")