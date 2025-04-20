import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("winter_2024_data.csv")
    return df

df = load_data()

st.title("🚗 CarMax Trade-in & Purchase Dashboard")
st.markdown("Explore customer behavior based on trade-ins and purchases at CarMax.")

# Sidebar filters
st.sidebar.header("Filter Options")
states = st.sidebar.multiselect("Select State(s)", options=df['state'].unique(), default=df['state'].unique())
make_filter = st.sidebar.multiselect("Select Appraised Make(s)", options=df['make_appraisal'].unique(), default=df['make_appraisal'].unique())

# Apply filters
filtered_df = df[(df['state'].isin(states)) & (df['make_appraisal'].isin(make_filter))]

# 🔦 Highlights
st.markdown("### 🔍 Dashboard Highlights")
colA, colB, colC = st.columns(3)
with colA:
    st.metric("Total Unique Appraised Makes", len(df['make_appraisal'].unique()))
with colB:
    st.metric("Total Unique Purchased Makes", len(df['make'].unique()))    
with colC:
    same_make_pct = (df['make'] == df['make_appraisal']).mean() * 100
    st.metric("% Customers Staying with Same Make", f"{same_make_pct:.2f}%")

st.divider()

# 1. Most common appraised vs purchased makes
st.subheader("1️⃣ Top Appraised vs Purchased Makes")

# Add comparison table
st.markdown("#### Comparison Table of Top 20 Appraised vs Purchased Makes")
top_appraised_full = filtered_df['make_appraisal'].value_counts().head(20).reset_index()
top_purchased_full = filtered_df['make'].value_counts().head(20).reset_index()
top_appraised_full.columns = ['Make', 'Appraised Count']
top_purchased_full.columns = ['Make', 'Purchased Count']
top_comparison = pd.merge(top_appraised_full, top_purchased_full, on='Make', how='outer').fillna(0)
top_comparison[['Appraised Count', 'Purchased Count']] = top_comparison[['Appraised Count', 'Purchased Count']].astype(int)
st.dataframe(top_comparison)

col1, col2 = st.columns(2)
with col1:
    top_appraised = filtered_df['make_appraisal'].value_counts().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots()
    sns.barplot(x=top_appraised.values, y=top_appraised.index, ax=ax)
    ax.set_title("Top 10 Appraised Makes")
    st.pyplot(fig)
    st.caption("This chart shows the 10 most frequently appraised car makes.")

with col2:
    top_purchased = filtered_df['make'].value_counts().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots()
    sns.barplot(x=top_purchased.values, y=top_purchased.index, ax=ax)
    ax.set_title("Top 10 Purchased Makes")
    st.pyplot(fig)
    st.caption("This chart shows the 10 most frequently purchased car makes.")


