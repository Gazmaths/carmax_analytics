import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
url = "https://github.com/Gazmaths/carmax_analytics/blob/main/winter_2024_data.csv"
df = pd.read_csv(url)

# Streamlit dashboard title
st.title("Vehicle Appraisal Dashboard")

# Display basic statistics
st.header("Basic Data Overview")
st.write(df.describe())

# Filter options for the user
st.sidebar.header("Filters")
selected_state = st.sidebar.selectbox("Select State", df['state'].unique())
selected_vehicle_type = st.sidebar.selectbox("Select Vehicle Type", df['vehicle_type'].unique())

# Filter data based on selections
filtered_data = df[(df['state'] == selected_state) & (df['vehicle_type'] == selected_vehicle_type)]

# Display filtered data
st.subheader(f"Filtered Data for {selected_state} and {selected_vehicle_type}")
st.write(filtered_data)

# Visualizations
st.header("Data Visualizations")

# Scatter plot for price vs. mileage
st.subheader("Price vs Mileage")
fig, ax = plt.subplots()
sns.scatterplot(data=filtered_data, x='mileage', y='price', ax=ax)
ax.set_title("Price vs Mileage")
st.pyplot(fig)

# Histogram for price distribution
st.subheader("Price Distribution")
fig, ax = plt.subplots()
sns.histplot(filtered_data['price'], kde=True, ax=ax)
ax.set_title("Price Distribution")
st.pyplot(fig)

# Boxplot for price based on trim level
st.subheader("Price by Trim Level")
fig, ax = plt.subplots()
sns.boxplot(data=filtered_data, x='trim_level', y='price', ax=ax)
ax.set_title("Price by Trim Level")
st.pyplot(fig)

# Show the correlation heatmap
st.subheader("Correlation Heatmap")
corr = filtered_data.corr()
fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
ax.set_title("Correlation Heatmap")
st.pyplot(fig)

