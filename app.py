import streamlit as st
import pandas as pd
import joblib
import json
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Employee Salary Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

#  CSS 
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .stSelectbox > div > div, .stNumberInput > div > div {
        background-color: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 10px;
    }
    .prediction-card {
        background: linear-gradient(135deg, #ff6b6b, #feca57);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        margin: 20px 0;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        height: 100%;
    }
    .title-gradient {
        background: linear-gradient(45deg, #ff6b6b, #feca57, #48dbfb, #ff9ff3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem; font-weight: bold; text-align: center; margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)


# Caching Functions
@st.cache_data
def load_model_and_preprocessors():
    """Load the trained model, encoders, and scaler."""
    try:
        data = joblib.load('salary_model.pkl')
        with open('category_mappings.json', 'r') as f:
            mappings = json.load(f)
        return data, mappings
    except FileNotFoundError:
        st.error("Model files not found! Please run trainmodel.py first.")
        st.stop()

@st.cache_data
def load_data(csv_path):
    """Load the original data for analysis."""
    try:
        df = pd.read_csv(csv_path)
        df.dropna(inplace=True)
        for col in ['Age', 'Years of Experience', 'Salary']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        return df
    except FileNotFoundError:
        st.error(f"Data file '{csv_path}' not found!")
        st.stop()

def create_data_visualization(df):
    """Create visualizations for the data analysis dashboard."""
    st.markdown("---")
    st.markdown("## 📊 Data Analysis Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Average Salary", f"${df['Salary'].mean():,.2f}")
    with col3:
        st.metric("Median Salary", f"${df['Salary'].median():,.2f}")
    with col4:
        st.metric("Salary Range", f"${df['Salary'].max() - df['Salary'].min():,.2f}")

    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.histogram(df, x='Salary', nbins=50, title='Salary Distribution', color_discrete_sequence=['#ff6b6b'])
        fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig1, use_container_width=True)
    
    with c2:
        fig2 = px.scatter(df, x='Years of Experience', y='Salary', color='Gender', title='Experience vs Salary',
                          color_discrete_map={'Male': '#48dbfb', 'Female': '#ff9ff3', 'Other': '#feca57'})
        fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig2, use_container_width=True)
    
    edu_salary = df.groupby('Education Level')['Salary'].mean().sort_values(ascending=False)
    fig3 = px.bar(
        x=edu_salary.index, 
        y=edu_salary.values, 
        title='Average Salary by Education Level', 
        color=edu_salary.values, 
        color_continuous_scale='viridis'
    )
    fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white', xaxis_title='Education Level', yaxis_title='Average Salary')
    st.plotly_chart(fig3, use_container_width=True)


# --- Main Application ---
def main():
    st.markdown('<h1 class="title-gradient">Employee Salary Predictor</h1>', unsafe_allow_html=True)

    model_data, category_mappings = load_model_and_preprocessors()
    df = load_data('Salary_Data.csv')

    with st.sidebar:
        st.markdown("##  Input Parameters")
        age = st.number_input("Age", 18, 100, 18)
        gender = st.selectbox("Gender", category_mappings['Gender']['classes'],placeholder="gender")
        education_level = st.selectbox("Education Level", category_mappings['Education Level']['classes'])
        job_title = st.selectbox("Job Title", category_mappings['Job Title']['classes'])
        years_experience = st.number_input("Years of Experience", 0.0, 50.0, 0.0, 0.5)
        
        predict_button = st.button("Predict Salary", type="primary", use_container_width=True)

    if predict_button:
        input_df = pd.DataFrame([{'Age': age, 'Gender': gender, 'Education Level': education_level, 'Job Title': job_title, 'Years of Experience': years_experience}])
        
        for col, le in model_data['label_encoders'].items():
            input_df[col] = le.transform(input_df[col])
            
        input_scaled = model_data['scaler'].transform(input_df)
        predicted_salary = model_data['model'].predict(input_scaled)[0]

        st.markdown(f'<div class="prediction-card"><h2>🎯 Predicted Salary</h2><h1>${predicted_salary:,.2f}</h1><p>Based on your profile and our best-performing AI model.</p></div>', unsafe_allow_html=True)
        
        # **UPDATED SECTION WITH MEDIAN CARD**
        c1, c2, c3 = st.columns(3)
        
        # Column 1: vs. Average
        with c1:
            avg_salary = df['Salary'].mean()
            diff_avg = predicted_salary - avg_salary
            st.markdown(f'<div class="metric-card"><h3>📊 vs. Average</h3><h2>${diff_avg:+,.2f}</h2><p>Dataset Average: ${avg_salary:,.2f}</p></div>', unsafe_allow_html=True)

        # Column 2: vs. Median (NEW)
        with c2:
            median_salary = df['Salary'].median()
            diff_median = predicted_salary - median_salary
            st.markdown(f'<div class="metric-card"><h3>🆚 vs. Median</h3><h2>${diff_median:+,.2f}</h2><p>Dataset Median: ${median_salary:,.2f}</p></div>', unsafe_allow_html=True)

        # Column 3: Percentile
        with c3:
            percentile = (df['Salary'] < predicted_salary).mean() * 100
            st.markdown(f'<div class="metric-card"><h3>📈 Percentile</h3><h2>Top {100-percentile:.1f}%</h2><p>Higher than {percentile:.1f}% of profiles.</p></div>', unsafe_allow_html=True)
    
    create_data_visualization(df) 
    
    st.markdown("---")
    st.markdown("## 🤖 About the AI Model")
    st.markdown("""
    This prediction is powered by a machine learning model. To ensure the highest accuracy, we trained and compared **8 different models**, including Linear Regression, Lasso, Ridge, Support Vector Machines, K-Neighbors, Random Forest, XGBoost, and Gradient Boosting. The best-performing model was automatically selected and deployed for this application.
    """)
    st.markdown("---")

    # --- UPDATED FOOTER SECTION ---
    st.markdown("""
    <div style="text-align: center; color: rgba(255, 255, 255, 0.7);">
        <p>Made by Dheemanth</p>
        <p>
            <a href="www.linkedin.com/in/dheemanth-dhanpalal-3a50bb324" target="_blank" style="color: #0077B5; margin: 0 10px; text-decoration: none;">LinkedIn</a> |
            <a href="https://github.com/Dheemanth49/Employee-Salary-Prediction.git" target="_blank" style="color: #FFFFFF; margin: 0 10px; text-decoration: none;">GitHub</a> |
            <a href="mailto:dheemanthdhanpal@gmail.com" style="color: #EA4335; margin: 0 10px; text-decoration: none;">Email</a>
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()