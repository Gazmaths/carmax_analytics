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
    st.caption("This chart shows the 10 most frequently appraised car makes.")

with col2:
    top_purchased = filtered_df['make'].value_counts().head(10)
    st.bar_chart(top_purchased)
    st.caption("This chart shows the 10 most frequently purchased car makes.")

# 2. Appraisal offer vs. purchase price
st.subheader("💰 Appraisal Offer vs. Purchase Price")
fig1, ax1 = plt.subplots()
sns.scatterplot(data=filtered_df, x='appraisal_offer', y='price', alpha=0.5, ax=ax1)
ax1.set_title("Appraisal Offer vs Purchase Price")
st.pyplot(fig1)
st.caption("This scatterplot compares the trade-in appraisal offer with the final purchase price, helping identify upgrade or downgrade behavior.")

# 3. Trade-in vs Purchase Vehicle Types
st.subheader("🚘 Vehicle Type Shift: Appraisal vs Purchase")
vehicle_shift = filtered_df.groupby(['vehicle_type_appraisal', 'vehicle_type']).size().unstack().fillna(0)
st.dataframe(vehicle_shift.style.format(precision=0))
st.caption("This table shows how vehicle types change from trade-in to purchase.")

# 4. Time between appraisal and purchase
st.subheader("⏳ Days Between Appraisal and Purchase")
fig2, ax2 = plt.subplots()
sns.histplot(filtered_df['days_since_offer'], bins=30, kde=True, ax=ax2)
ax2.set_title("Distribution of Days Between Appraisal and Purchase")
ax2.set_xlabel("Days")
ax2.set_ylabel("Frequency")
st.pyplot(fig2)
st.caption("This histogram shows how long customers typically wait between getting an appraisal and purchasing a vehicle.")

# 5. Online vs In-Person Appraisals
st.subheader("🧑‍💻 Online vs In-Person Appraisals")
appraisal_type_counts = filtered_df['online_appraisal_flag'].map({0.0: 'In-Person', 1.0: 'Online'}).value_counts()
st.bar_chart(appraisal_type_counts)
st.caption("This bar chart compares the number of appraisals done online vs. in-person.")

# 6. Bar showing distribution of appraised makes
st.subheader("📊 Distribution of Appraised Makes")
make_counts = df['make_appraisal'].value_counts().head(20)
st.bar_chart(make_counts)
st.caption("This chart shows the 20 most common makes of appraised vehicles.")

# 7. Distribution of models within a selected make
st.subheader("🚙 Distribution of Models within a Selected Make")
selected_make = st.selectbox("Choose a Make to See Model Distribution", df['make_appraisal'].unique())
model_counts = df[df['make_appraisal'] == selected_make]['model_appraisal'].value_counts().head(20)
st.bar_chart(model_counts)
st.caption(f"This chart shows the top 20 models appraised under the make: {selected_make}.")

# 8. Price vs Appraisal Offer by Model + Depreciation
st.subheader("📉 Price vs Appraisal Offer by Model (with Depreciation)")
df['depreciation_pct'] = ((df['appraisal_offer'] - df['price']) / df['appraisal_offer']) * 100
depreciation_df = df[['model_appraisal', 'appraisal_offer', 'price', 'depreciation_pct']].dropna()
top_models = depreciation_df['model_appraisal'].value_counts().head(10).index
dep_chart_data = depreciation_df[depreciation_df['model_appraisal'].isin(top_models)]
fig3, ax3 = plt.subplots(figsize=(10, 5))
sns.barplot(data=dep_chart_data, x='model_appraisal', y='depreciation_pct', ci=None, ax=ax3)
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
ax3.set_title("Average Depreciation Percentage by Appraised Model")
ax3.set_ylabel("Depreciation (%)")
st.pyplot(fig3)
st.caption("This bar chart shows how much value is typically lost between the appraisal offer and the purchase price, by model.")
