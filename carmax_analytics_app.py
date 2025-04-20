import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np
import plotly.express as px
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
# 1. Comparison Table of Top 10 Appraised vs Purchased Makes
st.subheader("1️⃣ Top Appraised vs Purchased Makes")
top_appraised_full = filtered_df['make_appraisal'].value_counts().sort_values(ascending=False).head(10).reset_index()
top_purchased_full = filtered_df['make'].value_counts().sort_values(ascending=False).head(10).reset_index()

top_appraised_full.columns = ['Make', 'Appraised Count']
top_purchased_full.columns = ['Make', 'Purchased Count']

top_comparison = pd.merge(top_appraised_full, top_purchased_full, on='Make', how='outer').fillna(0)

# Ensure the counts are integer values
top_comparison[['Appraised Count', 'Purchased Count']] = top_comparison[['Appraised Count', 'Purchased Count']].astype(int)

# Display the dataframe
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
st.divider()

# 2. Map visualization for appraised cars by state
st.subheader("2️⃣ Map of Appraised Cars by State")

# Group by state and count the number of appraisals
state_appraisal_count = filtered_df['state'].value_counts().reset_index()
state_appraisal_count.columns = ['State', 'Appraisal Count']

# Using Plotly Express to create a choropleth map
fig = px.choropleth(state_appraisal_count,
                    locations='State',
                    locationmode='USA-states',
                    color='Appraisal Count',
                    hover_name='State',
                    color_continuous_scale="Viridis",
                    labels={'Appraisal Count': 'Number of Appraisals'},
                    title="Map of Appraised Cars by State")
st.plotly_chart(fig)

# 3. Appraised Makes by Selected State - Head and Tail (Side by Side)
st.subheader("2️⃣ Appraised Makes in Selected State")
selected_state = st.selectbox("Choose a State to See Appraised Makes Distribution", df['state'].unique())
state_df = df[df['state'] == selected_state]
state_make_counts = state_df['make_appraisal'].value_counts()

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Top 10 Makes (Head)")
    top_makes = state_make_counts.head(10)
    fig1, ax1 = plt.subplots()
    sns.barplot(x=top_makes.values, y=top_makes.index, ax=ax1)
    ax1.set_title(f"Top 10 Appraised Makes in {selected_state}")
    st.pyplot(fig1)

with col2:
    st.markdown("#### Bottom 10 Makes (Tail)")
    tail_makes = state_make_counts.tail(10)
    fig2, ax2 = plt.subplots()
    sns.barplot(x=tail_makes.values, y=tail_makes.index, ax=ax2)
    ax2.set_title(f"Bottom 10 Appraised Makes in {selected_state}")
    st.pyplot(fig2)

st.caption("These charts display the most and least frequently appraised car makes in the selected state.")

st.divider()

# 4. Distribution of models within a selected make
st.subheader("4️⃣ Distribution of Models within a Selected Make")
selected_make = st.selectbox("Choose a Make to See Model Distribution", df['make_appraisal'].unique())
model_counts = df[df['make_appraisal'] == selected_make]['model_appraisal'].value_counts().sort_values(ascending=False).head(20)
fig, ax = plt.subplots()
sns.barplot(x=model_counts.values, y=model_counts.index, ax=ax)
ax.set_title(f"Top 20 Models under {selected_make}")
st.pyplot(fig)
st.caption(f"This chart shows the top 20 models appraised under the make: {selected_make}.")

st.divider()

# 5. Appraisal offer vs. purchase price
st.subheader("5️⃣ Appraisal Offer vs. Purchase Price")
fig1, ax1 = plt.subplots()
sns.scatterplot(data=filtered_df, x='appraisal_offer', y='price', hue='vehicle_type', alpha=0.6, ax=ax1)
ax1.set_title("Appraisal Offer vs Purchase Price")
st.pyplot(fig1)
st.caption("This scatterplot compares the trade-in appraisal offer with the final purchase price, helping identify upgrade or downgrade behavior.")

st.divider()

# 6. Trade-in vs Purchase Vehicle Types
st.subheader("6️⃣ Vehicle Type Shift: Appraisal vs Purchase")
vehicle_shift = filtered_df.groupby(['vehicle_type_appraisal', 'vehicle_type']).size().unstack().fillna(0)
st.dataframe(vehicle_shift.style.format(precision=0))
st.caption("This table shows how vehicle types change from trade-in to purchase.")

st.divider()

# 7. Time between appraisal and purchase
st.subheader("7️⃣ Days Between Appraisal and Purchase")
fig2, ax2 = plt.subplots()
sns.histplot(filtered_df['days_since_offer'], bins=30, kde=True, ax=ax2, color="skyblue")
ax2.set_title("Distribution of Days Between Appraisal and Purchase")
ax2.set_xlabel("Days")
ax2.set_ylabel("Frequency")
st.pyplot(fig2)
st.caption("This histogram shows how long customers typically wait between getting an appraisal and purchasing a vehicle.")

st.divider()

# 8. Online vs In-Person Appraisals
st.subheader("8️⃣ Online vs In-Person Appraisals")
appraisal_type_counts = filtered_df['online_appraisal_flag'].map({0.0: 'In-Person', 1.0: 'Online'}).value_counts()
fig, ax = plt.subplots()
sns.barplot(x=appraisal_type_counts.index, y=appraisal_type_counts.values, ax=ax)
ax.set_title("Online vs In-Person Appraisals")
ax.set_ylabel("Count")
st.pyplot(fig)
st.caption("This bar chart compares the number of appraisals done online vs. in-person.")

st.divider()

# 9. Price vs Appraisal Offer by Model + Depreciation
st.subheader("9️⃣ Price vs Appraisal Offer by Model (with Depreciation)")
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

st.divider()

# 10. Machine Learning Model to Predict Same-Make Purchases
st.subheader("🔟 Predicting Same-Make Purchases with Random Forest")
st.markdown("We use a Random Forest Classifier to predict whether a customer will purchase a car from the same make as their appraised vehicle.")

features = [
    'make_appraisal', 'model_appraisal', 'trim_level_appraisal', 'model_year_appraisal',
    'mileage_appraisal', 'engine_appraisal', 'cylinders_appraisal', 'mpg_city_appraisal',
    'mpg_highway_appraisal', 'horsepower_appraisal', 'fuel_capacity_appraisal',
    'vehicle_type_appraisal', 'color_appraisal'
]

# Create target
df['same_make'] = (df['make'] == df['make_appraisal']).astype(int)

# Prepare data
df_model = df[features + ['same_make']].dropna()
label_encoders = {}
for col in df_model.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    label_encoders[col] = le

X = df_model[features]
y = df_model['same_make']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)

st.write("**Classification Report**")
st.json(report)
st.caption("The model performs well in identifying customers who switch car makes, but has lower precision for those who stay with the same make. This insight could guide targeted marketing strategies.")


