import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from sklearn.preprocessing import StandardScaler
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(layout="wide", page_title="Parkinson's Tremor Analysis Dashboard")

# Dashboard title
st.title("Smart Glove for Parkinson's Tremor Analysis")
st.markdown("### Data Visualization and Analysis Dashboard")

# Sidebar for controls
st.sidebar.header("Controls")

# Function to load data
@st.cache_data
def load_data():
    # This is a placeholder - replace with your actual data loading code
    # For demo purposes, let's create synthetic data similar to PD tremor patterns
    
    # Create time array
    time = np.arange(0, 60, 0.01)
    
    # Create different tremor patterns
    
    # 1. Resting tremor (4-6 Hz oscillation)
    resting_tremor_acc_x = 0.5 * np.sin(2 * np.pi * 5 * time) + 0.2 * np.random.randn(len(time))
    resting_tremor_acc_y = 0.4 * np.sin(2 * np.pi * 5.2 * time + 0.5) + 0.2 * np.random.randn(len(time))
    resting_tremor_acc_z = 0.3 * np.sin(2 * np.pi * 4.8 * time + 1.0) + 0.2 * np.random.randn(len(time))
    
    resting_tremor_gyro_x = 0.4 * np.sin(2 * np.pi * 5 * time + 0.2) + 0.15 * np.random.randn(len(time))
    resting_tremor_gyro_y = 0.3 * np.sin(2 * np.pi * 5.2 * time + 0.7) + 0.15 * np.random.randn(len(time))
    resting_tremor_gyro_z = 0.25 * np.sin(2 * np.pi * 4.8 * time + 1.2) + 0.15 * np.random.randn(len(time))
    
    # 2. Postural tremor (higher frequency, 6-8 Hz)
    postural_tremor_acc_x = 0.8 * np.sin(2 * np.pi * 7 * time) + 0.3 * np.random.randn(len(time))
    postural_tremor_acc_y = 0.7 * np.sin(2 * np.pi * 7.2 * time + 0.5) + 0.3 * np.random.randn(len(time))
    postural_tremor_acc_z = 0.6 * np.sin(2 * np.pi * 6.8 * time + 1.0) + 0.3 * np.random.randn(len(time))
    
    postural_tremor_gyro_x = 0.7 * np.sin(2 * np.pi * 7 * time + 0.2) + 0.25 * np.random.randn(len(time))
    postural_tremor_gyro_y = 0.65 * np.sin(2 * np.pi * 7.2 * time + 0.7) + 0.25 * np.random.randn(len(time))
    postural_tremor_gyro_z = 0.6 * np.sin(2 * np.pi * 6.8 * time + 1.2) + 0.25 * np.random.randn(len(time))
    
    # 3. Kinetic tremor (more irregular, 4-10 Hz with varying amplitude)
    t_mod = np.linspace(0, 10, len(time))  # modulation factor
    kinetic_tremor_acc_x = 1.0 * np.sin(2 * np.pi * (5 + t_mod/2) * time) + 0.4 * np.random.randn(len(time))
    kinetic_tremor_acc_y = 0.9 * np.sin(2 * np.pi * (5.2 + t_mod/2) * time + 0.5) + 0.4 * np.random.randn(len(time))
    kinetic_tremor_acc_z = 0.8 * np.sin(2 * np.pi * (4.8 + t_mod/2) * time + 1.0) + 0.4 * np.random.randn(len(time))
    
    kinetic_tremor_gyro_x = 0.9 * np.sin(2 * np.pi * (5 + t_mod/2) * time + 0.2) + 0.35 * np.random.randn(len(time))
    kinetic_tremor_gyro_y = 0.85 * np.sin(2 * np.pi * (5.2 + t_mod/2) * time + 0.7) + 0.35 * np.random.randn(len(time))
    kinetic_tremor_gyro_z = 0.8 * np.sin(2 * np.pi * (4.8 + t_mod/2) * time + 1.2) + 0.35 * np.random.randn(len(time))
    
    # Create dataframes for each tremor type
    df_resting = pd.DataFrame({
        'Time': time,
        'Accel_X': resting_tremor_acc_x,
        'Accel_Y': resting_tremor_acc_y,
        'Accel_Z': resting_tremor_acc_z,
        'Gyro_X': resting_tremor_gyro_x,
        'Gyro_Y': resting_tremor_gyro_y,
        'Gyro_Z': resting_tremor_gyro_z,
        'Tremor_Type': 'Resting'
    })
    
    df_postural = pd.DataFrame({
        'Time': time,
        'Accel_X': postural_tremor_acc_x,
        'Accel_Y': postural_tremor_acc_y,
        'Accel_Z': postural_tremor_acc_z,
        'Gyro_X': postural_tremor_gyro_x,
        'Gyro_Y': postural_tremor_gyro_y,
        'Gyro_Z': postural_tremor_gyro_z,
        'Tremor_Type': 'Postural'
    })
    
    df_kinetic = pd.DataFrame({
        'Time': time,
        'Accel_X': kinetic_tremor_acc_x,
        'Accel_Y': kinetic_tremor_acc_y,
        'Accel_Z': kinetic_tremor_acc_z,
        'Gyro_X': kinetic_tremor_gyro_x,
        'Gyro_Y': kinetic_tremor_gyro_y,
        'Gyro_Z': kinetic_tremor_gyro_z,
        'Tremor_Type': 'Kinetic'
    })
    
    # Combine all data
    df = pd.concat([df_resting, df_postural, df_kinetic], ignore_index=True)
    
    # Calculate derived metrics
    df['Accel_Magnitude'] = np.sqrt(df['Accel_X']**2 + df['Accel_Y']**2 + df['Accel_Z']**2)
    df['Gyro_Magnitude'] = np.sqrt(df['Gyro_X']**2 + df['Gyro_Y']**2 + df['Gyro_Z']**2)
    
    # Add severity levels (for demonstration)
    # In a real implementation, this would be based on clinical criteria
    conditions = [
        (df['Accel_Magnitude'] < 0.6),
        (df['Accel_Magnitude'] >= 0.6) & (df['Accel_Magnitude'] < 1.0),
        (df['Accel_Magnitude'] >= 1.0)
    ]
    
    values = ['Mild', 'Moderate', 'Severe']
    df['Severity'] = np.select(conditions, values)
    
    # Add patient IDs for demonstration
    patient_ids = ['PD001', 'PD002', 'PD003']
    df['Patient_ID'] = np.random.choice(patient_ids, size=len(df))
    
    # Calculate tremor score using your formula
    df['Tremor_Score'] = (0.4 * df['Accel_Magnitude']) + (0.3 * df['Gyro_Magnitude']) + \
                          (0.2 * df[['Accel_X', 'Accel_Y', 'Accel_Z']].std(axis=1)) + \
                          (0.1 * np.random.uniform(0.1, 0.5, size=len(df)))  # Simulated HRV
    
    return df

# Load or generate data
df = load_data()

# Add filter controls
st.sidebar.subheader("Filter Data")
tremor_types = st.sidebar.multiselect("Tremor Type", options=df['Tremor_Type'].unique(), default=df['Tremor_Type'].unique())
severity_levels = st.sidebar.multiselect("Severity Level", options=df['Severity'].unique(), default=df['Severity'].unique())
patient_selection = st.sidebar.multiselect("Patient ID", options=df['Patient_ID'].unique(), default=df['Patient_ID'].unique()[0])

# Filter data based on selections
filtered_df = df[(df['Tremor_Type'].isin(tremor_types)) & 
                 (df['Severity'].isin(severity_levels)) & 
                 (df['Patient_ID'].isin(patient_selection))]

# Analysis options
st.sidebar.subheader("Analysis Options")
analysis_type = st.sidebar.selectbox("Analysis Type", 
                                     ["Raw Signal Analysis", 
                                      "Frequency Domain Analysis", 
                                      "Tremor Score Analysis", 
                                      "Comparative Analysis"])

# Main content
if filtered_df.empty:
    st.warning("No data available with the selected filters.")
else:
    # Display different analyses based on selection
    if analysis_type == "Raw Signal Analysis":
        st.header("Raw Signal Analysis")
        
        # Sensor selection
        sensor_type = st.selectbox("Select Sensor Type", ["Accelerometer", "Gyroscope"])
        axis = st.selectbox("Select Axis", ["X", "Y", "Z", "Magnitude"])
        
        # Time range slider
        min_time = filtered_df['Time'].min()
        max_time = filtered_df['Time'].max()
        time_range = st.slider("Time Range (seconds)", 
                              min_value=float(min_time), 
                              max_value=float(max_time),
                              value=(float(min_time), min(float(min_time + 5), float(max_time))))
        
        # Filter by time range
        time_filtered_df = filtered_df[(filtered_df['Time'] >= time_range[0]) & (filtered_df['Time'] <= time_range[1])]
        
        # Create visualization
        if axis == "Magnitude":
            if sensor_type == "Accelerometer":
                y_col = "Accel_Magnitude"
                title = "Accelerometer Magnitude"
            else:
                y_col = "Gyro_Magnitude"
                title = "Gyroscope Magnitude"
        else:
            if sensor_type == "Accelerometer":
                y_col = f"Accel_{axis}"
                title = f"Accelerometer {axis}-axis"
            else:
                y_col = f"Gyro_{axis}"
                title = f"Gyroscope {axis}-axis"
        
        # Create plot
        fig = px.line(time_filtered_df, x="Time", y=y_col, color="Tremor_Type", 
                      title=f"{title} Raw Signal", 
                      labels={y_col: f"{sensor_type} Reading"})
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Signal statistics
        st.subheader("Signal Statistics")
        cols = st.columns(4)
        
        cols[0].metric("Mean", f"{time_filtered_df[y_col].mean():.3f}")
        cols[1].metric("Std Dev", f"{time_filtered_df[y_col].std():.3f}")
        cols[2].metric("Min", f"{time_filtered_df[y_col].min():.3f}")
        cols[3].metric("Max", f"{time_filtered_df[y_col].max():.3f}")
        
        # 3D visualization for all axes
        st.subheader("3D Visualization")
        
        # Take a sample of points for better visualization
        sample_size = min(1000, len(time_filtered_df))
        sampled_df = time_filtered_df.sample(sample_size) if len(time_filtered_df) > sample_size else time_filtered_df
        
        if sensor_type == "Accelerometer":
            fig_3d = px.scatter_3d(sampled_df, x='Accel_X', y='Accel_Y', z='Accel_Z',
                                   color='Tremor_Type', opacity=0.7,
                                   title="3D Accelerometer Readings")
        else:
            fig_3d = px.scatter_3d(sampled_df, x='Gyro_X', y='Gyro_Y', z='Gyro_Z',
                                   color='Tremor_Type', opacity=0.7,
                                   title="3D Gyroscope Readings")
            
        st.plotly_chart(fig_3d, use_container_width=True)

    elif analysis_type == "Frequency Domain Analysis":
        st.header("Frequency Domain Analysis")
        
        # Sensor selection
        sensor_type = st.selectbox("Select Sensor Type", ["Accelerometer", "Gyroscope"])
        axis = st.selectbox("Select Axis", ["X", "Y", "Z", "Magnitude"])
        
        # Function to compute FFT
        def compute_fft(data, fs=100):  # Assuming 100 Hz sampling rate
            # Ensure data is evenly sampled by creating a continuous time index
            n = len(data)
            yf = np.fft.rfft(data)
            xf = np.fft.rfftfreq(n, 1/fs)
            return xf, np.abs(yf)
        
        # Get column name based on selection
        if axis == "Magnitude":
            if sensor_type == "Accelerometer":
                y_col = "Accel_Magnitude"
            else:
                y_col = "Gyro_Magnitude"
        else:
            if sensor_type == "Accelerometer":
                y_col = f"Accel_{axis}"
            else:
                y_col = f"Gyro_{axis}"
        
        # Create subplots for each tremor type
        fig = make_subplots(rows=1, cols=len(tremor_types), 
                            subplot_titles=[f"{t} Tremor" for t in tremor_types],
                            shared_yaxes=True)
        
        for i, tremor in enumerate(tremor_types):
            tremor_data = filtered_df[filtered_df['Tremor_Type'] == tremor][y_col].values
            
            # Compute FFT
            xf, yf = compute_fft(tremor_data)
            
            # Only show frequencies up to 15 Hz (typical range for tremor analysis)
            mask = xf <= 15
            
            # Add trace to subplot
            fig.add_trace(
                go.Scatter(x=xf[mask], y=yf[mask], name=tremor),
                row=1, col=i+1
            )
            
            # Add peak frequency annotation
            peak_idx = np.argmax(yf[mask])
            peak_freq = xf[mask][peak_idx]
            
            fig.add_annotation(
                x=peak_freq,
                y=yf[mask][peak_idx],
                text=f"Peak: {peak_freq:.1f} Hz",
                showarrow=True,
                arrowhead=1,
                row=1, col=i+1
            )
        
        fig.update_layout(
            title=f"Frequency Domain Analysis: {sensor_type} {axis}",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Amplitude",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Frequency band analysis
        st.subheader("Frequency Band Analysis")
        
        def frequency_band_power(x, y):
            # Define bands commonly used for tremor analysis
            bands = {
                "1-3 Hz": (1, 3),
                "3-5 Hz": (3, 5),
                "5-7 Hz": (5, 7),
                "7-12 Hz": (7, 12)
            }
            
            powers = {}
            for band_name, (low, high) in bands.items():
                # Find indices corresponding to frequency range
                indices = (x >= low) & (x <= high)
                # Calculate power in band
                powers[band_name] = np.sum(y[indices])
                
            return powers
        
        # Calculate band powers for each tremor type
        band_powers = {}
        for tremor in tremor_types:
            tremor_data = filtered_df[filtered_df['Tremor_Type'] == tremor][y_col].values
            xf, yf = compute_fft(tremor_data)
            band_powers[tremor] = frequency_band_power(xf, yf)
        
        # Create a dataframe for visualization
        band_df = pd.DataFrame(band_powers).reset_index().rename(columns={'index': 'Frequency Band'})
        
        # Plot frequency band powers
        fig_bands = px.bar(band_df.melt(id_vars=['Frequency Band'], 
                                        var_name='Tremor Type', 
                                        value_name='Power'),
                          x='Frequency Band', y='Power', color='Tremor Type',
                          barmode='group',
                          title="Power Distribution Across Frequency Bands")
        
        st.plotly_chart(fig_bands, use_container_width=True)
        
        st.markdown("""
        ### Frequency Analysis Insights:

        - **Resting tremors** typically have dominant frequencies in the 4-6 Hz range
        - **Postural tremors** often show higher frequencies (6-8 Hz)
        - **Kinetic tremors** display broader frequency distributions and more irregularity
        
        The Smart Glove algorithm uses these frequency characteristics in the tremor classification process.
        """)

    elif analysis_type == "Tremor Score Analysis":
        st.header("Tremor Score Analysis")
        
        # Visualization of Tremor Score distribution
        fig_hist = px.histogram(filtered_df, x="Tremor_Score", color="Tremor_Type", 
                               barmode="overlay", opacity=0.7,
                               title="Tremor Score Distribution by Tremor Type",
                               marginal="box")
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Tremor Score over time
        st.subheader("Tremor Score Over Time")
        
        # Time range slider
        min_time = filtered_df['Time'].min()
        max_time = filtered_df['Time'].max()
        time_range = st.slider("Time Range (seconds)", 
                              min_value=float(min_time), 
                              max_value=float(max_time),
                              value=(float(min_time), min(float(min_time + 10), float(max_time))),
                              key="time_range_2")
        
        # Filter by time range
        time_filtered_df = filtered_df[(filtered_df['Time'] >= time_range[0]) & (filtered_df['Time'] <= time_range[1])]
        
        # Create plot with threshold lines
        fig_score = px.line(time_filtered_df, x="Time", y="Tremor_Score", color="Tremor_Type",
                           title="Tremor Score Time Series")
        
        # Add threshold lines
        fig_score.add_hline(y=0.5, line_dash="dash", line_color="green", annotation_text="Mild Tremor Threshold")
        fig_score.add_hline(y=1.0, line_dash="dash", line_color="orange", annotation_text="Moderate Tremor Threshold")
        fig_score.add_hline(y=1.5, line_dash="dash", line_color="red", annotation_text="Severe Tremor Threshold")
        
        st.plotly_chart(fig_score, use_container_width=True)
        
        # Correlation between components and tremor score
        st.subheader("Component Contribution to Tremor Score")
        
        # Calculate components of tremor score
        score_df = filtered_df.copy()
        score_df['Accel_Component'] = 0.4 * score_df['Accel_Magnitude']
        score_df['Gyro_Component'] = 0.3 * score_df['Gyro_Magnitude']
        score_df['FSR_Component'] = 0.2 * score_df[['Accel_X', 'Accel_Y', 'Accel_Z']].std(axis=1)  # Using accelerometer std as proxy
        score_df['HRV_Component'] = score_df['Tremor_Score'] - score_df['Accel_Component'] - score_df['Gyro_Component'] - score_df['FSR_Component']
        
        # Create stacked area chart
        fig_components = px.area(
            score_df,
            x="Time",
            y=["Accel_Component", "Gyro_Component", "FSR_Component", "HRV_Component"],
            title="Tremor Score Components Over Time",
            labels={"value": "Component Value", "variable": "Component"},
            color_discrete_map={
                "Accel_Component": "blue",
                "Gyro_Component": "green",
                "FSR_Component": "orange",
                "HRV_Component": "red"
            }
        )
        
        # Sample points for better visualization
        time_points = np.linspace(min_time, min(min_time + 5, max_time), 50)
        sampled_score_df = score_df[score_df['Time'].isin(time_points)]
        
        fig_components.update_layout(height=500)
        st.plotly_chart(fig_components, use_container_width=True)
        
        # Pie chart of average component contribution
        component_cols = ['Accel_Component', 'Gyro_Component', 'FSR_Component', 'HRV_Component']
        component_means = score_df[component_cols].mean()
        
        fig_pie = px.pie(
            values=component_means,
            names=component_means.index,
            title="Average Component Contribution to Tremor Score",
            color=component_means.index,
            color_discrete_map={
                "Accel_Component": "blue",
                "Gyro_Component": "green",
                "FSR_Component": "orange",
                "HRV_Component": "red"
            }
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.markdown("""
        ### Tremor Score Insights:
        
        The Smart Glove's Tremor Score is calculated using a weighted formula:
        
        **TremorScore = (0.4 × AccelRMS) + (0.3 × GyroRMS) + (0.2 × FSRStdDev) + (0.1 × HRV)**
        
        This analysis shows how each component contributes to the overall tremor assessment:
        
        - **Accelerometer data** (40% weight) provides the primary tremor intensity measurement
        - **Gyroscope data** (30% weight) captures rotational movements characteristic of different tremor types
        - **Force sensing** (20% weight) detects grip variations during tremors
        - **Heart rate variability** (10% weight) accounts for physiological stress factors
        
        The threshold lines at 0.5, 1.0, and 1.5 represent the boundaries between mild, moderate, and severe tremor classifications.
        """)

    elif analysis_type == "Comparative Analysis":
        st.header("Comparative Analysis")
        
        # Select comparison parameters
        col1, col2 = st.columns(2)
        
        with col1:
            x_param = st.selectbox("X-axis Parameter", 
                                  ["Accel_Magnitude", "Gyro_Magnitude", "Tremor_Score"],
                                  index=0)
            
        with col2:
            y_param = st.selectbox("Y-axis Parameter", 
                                  ["Accel_Magnitude", "Gyro_Magnitude", "Tremor_Score"],
                                  index=2)
            
        # Create scatter plot
        fig_scatter = px.scatter(filtered_df, x=x_param, y=y_param, 
                                color="Tremor_Type", facet_col="Severity",
                                opacity=0.7, title=f"{y_param} vs {x_param} by Tremor Type and Severity",
                                trendline="ols")
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Summary statistics by tremor type and severity
        st.subheader("Summary Statistics")
        
        # Group by tremor type and severity
        grouped_stats = filtered_df.groupby(['Tremor_Type', 'Severity']).agg({
            'Accel_Magnitude': ['mean', 'std', 'min', 'max'],
            'Gyro_Magnitude': ['mean', 'std', 'min', 'max'],
            'Tremor_Score': ['mean', 'std', 'min', 'max']
        }).round(3)
        
        st.dataframe(grouped_stats, use_container_width=True)
        
        # Radar chart of characteristics
        st.subheader("Tremor Type Characteristic Comparison")
        
        # Calculate metrics for radar chart
        radar_metrics = filtered_df.groupby('Tremor_Type').agg({
            'Accel_Magnitude': 'mean',
            'Gyro_Magnitude': 'mean',
            'Tremor_Score': 'mean'
        })
        
        # Add frequency metrics (simulated)
        radar_metrics['Frequency'] = [5.0, 7.0, 6.5]  # Dominant frequencies
        radar_metrics['Variability'] = [0.3, 0.5, 0.8]  # Signal variability
        
        # Normalize metrics for radar chart
        scaler = StandardScaler()
        radar_metrics_scaled = pd.DataFrame(
            scaler.fit_transform(radar_metrics), 
            columns=radar_metrics.columns,
            index=radar_metrics.index
        )
        
        # Create radar chart
        categories = radar_metrics_scaled.columns.tolist()
        fig_radar = go.Figure()
        
        for tremor_type in radar_metrics_scaled.index:
            fig_radar.add_trace(go.Scatterpolar(
                r=radar_metrics_scaled.loc[tremor_type].values.tolist() + [radar_metrics_scaled.loc[tremor_type].values.tolist()[0]],
                theta=categories + [categories[0]],
                fill='toself',
                name=tremor_type
            ))
            
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[-2, 2]
                )),
            title="Characteristic Profile by Tremor Type"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        
        st.markdown("""
        ### Comparative Analysis Insights:
        
        This analysis reveals characteristic patterns for each tremor type:
        
        - **Resting tremors** typically show lower amplitude but consistent frequency patterns
        - **Postural tremors** are characterized by higher gyroscope readings (rotational movement)  
        - **Kinetic tremors** display the greatest variability and highest overall tremor scores
        
        These distinct patterns enable the Smart Glove to accurately classify tremor types in real-time,
        providing valuable clinical information about tremor progression and response to medication.
        """)

st.sidebar.markdown("""
---
### About This Dashboard

This dashboard demonstrates the data analysis capabilities of the **Smart Glove for Parkinson's Tremor Detection** project.

The visualization and analysis tools shown here will be integrated into the final monitoring system, allowing clinicians to:

1. Monitor tremor patterns over time
2. Assess medication effectiveness 
3. Quantify disease progression
4. Distinguish between tremor types

**Technical Note:** Replace the simulated data with your actual dataset for your final presentation.
""")

# Download buttons for report generation
st.sidebar.markdown("---")
st.sidebar.subheader("Export Options")

if st.sidebar.button("Generate PDF Report"):
    st.sidebar.success("Report generation feature will be implemented in the final version.")

if st.sidebar.button("Export Data"):
    st.sidebar.success("Data export feature will be implemented in the final version.")
