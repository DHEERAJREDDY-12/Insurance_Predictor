import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from typing import Dict, Any
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Page configuration
st.set_page_config(
    page_title="Insurance Cost Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .metric-card h4 {
        color: #000000 !important;
        background-color: transparent !important;
    }
    .metric-card h2 {
        color: #000000 !important;
        background-color: transparent !important;
    }
    .metric-card p {
        color: #000000 !important;
        background-color: transparent !important;
    }
    .prediction-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
        text-align: center;
    }
    .prediction-value {
        font-size: 3.5rem;
        font-weight: bold;
        color: #000000 !important;
        margin-bottom: 0.5rem;
        background-color: transparent !important;
    }
    .prediction-label {
        font-size: 1.2rem;
        color: #000000 !important;
        background-color: transparent !important;
    }
    .prediction-box h2 {
        color: #000000 !important;
        background-color: transparent !important;
    }
    .prediction-box p {
        color: #000000 !important;
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

def load_model_and_metrics():
    """Load the trained model and metrics"""
    try:
        model = joblib.load("artifacts/best_model.joblib")
        with open("artifacts/metrics.json", "r") as f:
            metrics = json.load(f)
        return model, metrics
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.info("Please run the training script first: `python src/train.py`")
        return None, None

def predict_single_record(model, data: Dict[str, Any]) -> float:
    """Make prediction for a single record"""
    try:
        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]
        return float(prediction)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def predict_batch(model, df: pd.DataFrame) -> pd.DataFrame:
    """Make predictions for batch data"""
    try:
        predictions = model.predict(df)
        result_df = df.copy()
        result_df['predicted_charges'] = predictions
        return result_df
    except Exception as e:
        st.error(f"Batch prediction error: {e}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Insurance Cost Prediction</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model and metrics
    model, metrics = load_model_and_metrics()
    
    if model is None or metrics is None:
        st.stop()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Home", "📊 Model Performance", "🔮 Single Prediction", "📁 Batch Prediction", "📈 Data Explorer"]
    )
    
    if page == "🏠 Home":
        show_home_page(metrics)
    elif page == "📊 Model Performance":
        show_model_performance(metrics)
    elif page == "🔮 Single Prediction":
        show_single_prediction(model)
    elif page == "📁 Batch Prediction":
        show_batch_prediction(model)
    elif page == "📈 Data Explorer":
        show_data_explorer()

def show_home_page(metrics):
    """Display the home page with overview"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Welcome to Insurance Cost Prediction System
        
        This application uses machine learning to predict insurance costs based on various factors such as:
        - **Age**: Age of the person
        - **Sex**: Gender (male/female)
        - **BMI**: Body Mass Index
        - **Children**: Number of children
        - **Smoker**: Smoking status (yes/no)
        - **Region**: Geographic region
        
        ### How to Use:
        1. **Single Prediction**: Enter individual details to get a cost prediction
        2. **Batch Prediction**: Upload a CSV file with multiple records
        3. **Data Explorer**: Explore the dataset and model performance
        """)
    
    with col2:
        st.markdown("### Model Performance")
        st.markdown(f"""
        <div class="metric-card">
            <h4>Best Model: {metrics['best_model'].replace('_', ' ').title()}</h4>
            <p><strong>R² Score:</strong> {metrics['r2']:.3f}</p>
            <p><strong>RMSE:</strong> ${metrics['rmse']:,.0f}</p>
            <p><strong>MAE:</strong> ${metrics['mae']:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)

def show_model_performance(metrics):
    """Display model performance metrics"""
    st.title("📊 Model Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="R² Score (Accuracy)",
            value=f"{metrics['r2']:.3f}",
            help="Proportion of variance explained by the model"
        )
    
    with col2:
        st.metric(
            label="RMSE",
            value=f"${metrics['rmse']:,.0f}",
            help="Root Mean Square Error - average prediction error"
        )
    
    with col3:
        st.metric(
            label="MAE",
            value=f"${metrics['mae']:,.0f}",
            help="Mean Absolute Error - average absolute prediction error"
        )
    
    # Performance interpretation
    st.markdown("### Performance Interpretation")
    
    r2_score = metrics['r2']
    if r2_score >= 0.9:
        performance_level = "Excellent"
        color = "green"
    elif r2_score >= 0.8:
        performance_level = "Good"
        color = "blue"
    elif r2_score >= 0.7:
        performance_level = "Fair"
        color = "orange"
    else:
        performance_level = "Poor"
        color = "red"
    
    st.markdown(f"""
    <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; border-left: 5px solid {color};">
        <h4 style="color: #000000 !important; background-color: transparent !important;">Model Performance: {performance_level}</h4>
        <p style="color: #000000 !important; background-color: transparent !important;">The model explains <strong style="color: #000000 !important; background-color: transparent !important;">{r2_score:.1%}</strong> of the variance in insurance costs.</p>
        <p style="color: #000000 !important; background-color: transparent !important;">On average, predictions are within <strong style="color: #000000 !important; background-color: transparent !important;">${metrics['mae']:,.0f}</strong> of actual costs.</p>
    </div>
    """, unsafe_allow_html=True)

def show_single_prediction(model):
    """Display single prediction interface"""
    st.title("🔮 Single Prediction")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Enter Patient Details")
        
        age = st.slider("Age", min_value=18, max_value=100, value=30)
        sex = st.selectbox("Sex", ["male", "female"])
        bmi = st.slider("BMI", min_value=15.0, max_value=50.0, value=25.0, step=0.1)
        children = st.slider("Number of Children", min_value=0, max_value=10, value=0)
        smoker = st.selectbox("Smoker", ["no", "yes"])
        region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
        
        if st.button("Predict Insurance Cost", type="primary"):
            data = {
                "age": age,
                "sex": sex,
                "bmi": bmi,
                "children": children,
                "smoker": smoker,
                "region": region
            }
            
            prediction = predict_single_record(model, data)
            if prediction is not None:
                with col2:
                    st.subheader("Prediction Result")
                    st.markdown(f"""
                    <div class="prediction-box">
                        <div class="prediction-value">${prediction:,.2f}</div>
                        <div class="prediction-label">Predicted Insurance Cost</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show input summary
                    st.markdown("### Input Summary")
                    for key, value in data.items():
                        st.write(f"**{key.title()}:** {value}")

def show_batch_prediction(model):
    """Display batch prediction interface"""
    st.title("📁 Batch Prediction")
    
    st.markdown("Upload a CSV file with the following columns: `age`, `sex`, `bmi`, `children`, `smoker`, `region`")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(df)} records")
            
            # Show preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Check required columns
            required_cols = ["age", "sex", "bmi", "children", "smoker", "region"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
            else:
                if st.button("Generate Predictions", type="primary"):
                    with st.spinner("Generating predictions..."):
                        result_df = predict_batch(model, df)
                        
                        if result_df is not None:
                            st.success("Predictions generated successfully!")
                            
                            # Display results
                            st.subheader("Predictions")
                            st.dataframe(result_df)
                            
                            # Download button
                            csv = result_df.to_csv(index=False)
                            st.download_button(
                                label="Download Predictions CSV",
                                data=csv,
                                file_name="insurance_predictions.csv",
                                mime="text/csv"
                            )
                            
                            # Summary statistics
                            st.subheader("Prediction Summary")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Total Records", len(result_df))
                            with col2:
                                st.metric("Average Prediction", f"${result_df['predicted_charges'].mean():,.2f}")
                            with col3:
                                st.metric("Max Prediction", f"${result_df['predicted_charges'].max():,.2f}")
                            
                            # Distribution plot
                            fig = px.histogram(
                                result_df, 
                                x='predicted_charges',
                                title="Distribution of Predicted Insurance Costs",
                                labels={'predicted_charges': 'Predicted Cost ($)'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error reading file: {e}")

def show_data_explorer():
    """Display data exploration interface"""
    st.title("📈 Data Explorer")
    
    try:
        # Load the original dataset
        df = pd.read_csv("insurance.csv")
        st.success(f"Loaded dataset with {len(df)} records")
        
        # Basic statistics
        st.subheader("Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Features", len(df.columns) - 1)
        with col3:
            st.metric("Average Cost", f"${df['charges'].mean():,.2f}")
        with col4:
            st.metric("Max Cost", f"${df['charges'].max():,.2f}")
        
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(df.head(10))
        
        # Distribution plots
        col1, col2 = st.columns(2)
        
        with col1:
            # Age distribution
            fig_age = px.histogram(df, x='age', title="Age Distribution")
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            # BMI distribution
            fig_bmi = px.histogram(df, x='bmi', title="BMI Distribution")
            st.plotly_chart(fig_bmi, use_container_width=True)
        
        # Cost analysis
        st.subheader("Insurance Cost Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cost by smoker status
            smoker_costs = df.groupby('smoker')['charges'].mean().reset_index()
            fig_smoker = px.bar(smoker_costs, x='smoker', y='charges', 
                              title="Average Cost by Smoking Status")
            st.plotly_chart(fig_smoker, use_container_width=True)
        
        with col2:
            # Cost by region
            region_costs = df.groupby('region')['charges'].mean().reset_index()
            fig_region = px.bar(region_costs, x='region', y='charges',
                              title="Average Cost by Region")
            st.plotly_chart(fig_region, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Feature Correlations")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        fig_corr = px.imshow(corr_matrix, 
                           text_auto=True, 
                           aspect="auto",
                           title="Correlation Matrix")
        st.plotly_chart(fig_corr, use_container_width=True)
        
    except FileNotFoundError:
        st.error("Dataset file 'insurance.csv' not found. Please ensure the file exists in the project directory.")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")

if __name__ == "__main__":
    main()
