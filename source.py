import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart Glove Tremor Visualization", layout="wide")

# Function to load or generate dummy data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data.csv')
    except:
        st.warning("No data found, generating dummy data for demo!")
        np.random.seed(42)
        df = pd.DataFrame({
            'Time': np.arange(0, 100, 0.1),
            'accel_x': np.random.normal(0, 0.5, 1000),
            'accel_y': np.random.normal(0, 0.5, 1000),
            'accel_z': np.random.normal(0, 0.5, 1000),
            'gyro_x': np.random.normal(0, 0.5, 1000),
            'gyro_y': np.random.normal(0, 0.5, 1000),
            'gyro_z': np.random.normal(0, 0.5, 1000),
        })

    # Calculate Tremor Score
    df['tremor_score'] = np.sqrt(df['accel_x']**2 + df['accel_y']**2 + df['accel_z']**2)

    # Define severity
    conditions = [
        (df['tremor_score'] < 1.0),
        (df['tremor_score'] >= 1.0) & (df['tremor_score'] < 2.0),
        (df['tremor_score'] >= 2.0)
    ]
    values = ['Mild', 'Moderate', 'Severe']

    df['Severity'] = np.select(conditions, values, default='Unknown')

    return df

# Load data
df = load_data()

# Title
st.title("Smart Glove Tremor Data Visualization Dashboard")

# Show Data
st.subheader("Sample Data")
st.dataframe(df.head(50))

# Line plot of tremor score over time
st.subheader("Tremor Score Over Time")
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(df['Time'], df['tremor_score'], color='blue')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Tremor Score')
ax.set_title('Tremor Score vs Time')
st.pyplot(fig)

# Severity Distribution
st.subheader("Severity Distribution")
fig2, ax2 = plt.subplots(figsize=(8,6))
sns.countplot(x='Severity', data=df, palette='Set2', ax=ax2)
ax2.set_title('Tremor Severity Counts')
st.pyplot(fig2)

# Future Scope Section
st.markdown("""
### Future Scope
- Real-time streaming from Smart Glove over Bluetooth
- Tremor event detection and alert generation
- Longitudinal tremor tracking and trend analysis
- Integration with doctor's portal for remote patient monitoring
""")
