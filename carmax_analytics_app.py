import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    top_appraised = filtered_df['make_appraisal'].value_counts().head(10)
    st.bar_chart(top_appraised)
    st.caption("Top 10 Appraised Makes")

with col2:
    top_purchased = filtered_df['make'].value_counts().head(10)
    st.bar_chart(top_purchased)
    st.caption("Top 10 Purchased Makes")

# 2. Appraisal offer vs. purchase price
st.subheader("💰 Appraisal Offer vs. Purchase Price")
fig1, ax1 = plt.subplots()
sns.scatterplot(data=filtered_df, x='appraisal_offer', y='price', alpha=0.5, ax=ax1)
ax1.set_title("Appraisal Offer vs Purchase Price")
st.pyplot(fig1)

# 3. Trade-in vs Purchase Vehicle Types
st.subheader("🚘 Vehicle Type Shift: Appraisal vs Purchase")
vehicle_shift = filtered_df.groupby(['vehicle_type_appraisal', 'vehicle_type']).size().unstack().fillna(0)
st.dataframe(vehicle_shift.style.format(precision=0))

# 4. Time between appraisal and purchase
st.subheader("⏳ Days Between Appraisal and Purchase")
fig2, ax2 = plt.subplots()
sns.histplot(filtered_df['days_since_offer'], bins=30, kde=True, ax=ax2)
ax2.set_title("Distribution of Days Between Appraisal and Purchase")
ax2.set_xlabel("Days")
ax2.set_ylabel("Frequency")
st.pyplot(fig2)

# 5. Online vs In-Person Appraisals
st.subheader("🧑‍💻 Online vs In-Person Appraisals")
appraisal_type_counts = filtered_df['online_appraisal_flag'].map({0.0: 'In-Person', 1.0: 'Online'}).value_counts()
st.bar_chart(appraisal_type_counts)
st.caption("Distribution of Online vs In-Person Appraisals")
