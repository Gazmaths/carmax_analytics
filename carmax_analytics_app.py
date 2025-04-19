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

# 1. Most common appraised vs purchased makes
st.subheader("🔁 Top Appraised vs Purchased Makes")
col1, col2 = st.columns(2)
with col1:
    top_appraised = filtered_df['make_appraisal'].value_counts().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots()
    sns.barplot(x=top_appraised.values, y=top_appraised.index, palette="crest", ax=ax)
    ax.set_title("Top 10 Appraised Makes")
    st.pyplot(fig)
    st.caption("This chart shows the 10 most frequently appraised car makes.")

with col2:
    top_purchased = filtered_df['make'].value_counts().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots()
    sns.barplot(x=top_purchased.values, y=top_purchased.index, palette="flare", ax=ax)
    ax.set_title("Top 10 Purchased Makes")
    st.pyplot(fig)
    st.caption("This chart shows the 10 most frequently purchased car makes.")

# 6. Bar showing distribution of appraised makes
st.subheader("📊 Distribution of Appraised Makes")
make_counts = df['make_appraisal'].value_counts().sort_values(ascending=False).head(20)
fig, ax = plt.subplots()
sns.barplot(x=make_counts.values, y=make_counts.index, palette="mako", ax=ax)
ax.set_title("Distribution of Top 20 Appraised Makes")
st.pyplot(fig)
st.caption("This chart shows the 20 most common makes of appraised vehicles.")

# 7. Distribution of models within a selected make
st.subheader("🚙 Distribution of Models within a Selected Make")
selected_make = st.selectbox("Choose a Make to See Model Distribution", df['make_appraisal'].unique())
model_counts = df[df['make_appraisal'] == selected_make]['model_appraisal'].value_counts().sort_values(ascending=False).head(20)
fig, ax = plt.subplots()
sns.barplot(x=model_counts.values, y=model_counts.index, palette="light:#5A9", ax=ax)
ax.set_title(f"Top 20 Models under {selected_make}")
st.pyplot(fig)
st.caption(f"This chart shows the top 20 models appraised under the make: {selected_make}.")


