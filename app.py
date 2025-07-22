import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #e6e6e6;
        margin: 1rem 0;
        text-align: center;
    }
    .high-income {
        background-color: #d4edda;
        border-color: #28a745;
        color: #155724;
    }
    .low-income {
        background-color: #f8d7da;
        border-color: #dc3545;
        color: #721c24;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model_and_scaler():
    """Load the trained model and scaler"""
    try:
        model = joblib.load('IBM/best_model.pkl')
        scaler = joblib.load('IBM/scaler.pkl')
        feature_names = joblib.load('IBM/feature_names.pkl')
        return model, scaler, feature_names
    except FileNotFoundError:
        st.error("Model files not found! Please run train_model.py first.")
        st.stop()

def create_input_features():
    """Create input widgets for all features"""
    st.sidebar.header("📊 Employee Information")
    
    # Personal Information
    st.sidebar.subheader("Personal Details")
    age = st.sidebar.slider("Age", min_value=17, max_value=75, value=35)
    
    # Work Information
    st.sidebar.subheader("Work Details")
    workclass = st.sidebar.selectbox("Work Class", [
        "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
        "Local-gov", "State-gov", "Others"
    ])
    
    fnlwgt = st.sidebar.number_input("Final Weight", min_value=10000, max_value=1500000, value=200000)
    
    educational_num = st.sidebar.slider("Education Level (Years)", min_value=1, max_value=16, value=10)
    
    marital_status = st.sidebar.selectbox("Marital Status", [
        "Married-civ-spouse", "Divorced", "Never-married", "Separated",
        "Widowed", "Married-spouse-absent", "Married-AF-spouse"
    ])
    
    occupation = st.sidebar.selectbox("Occupation", [
        "Tech-support", "Craft-repair", "Other-service", "Sales",
        "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
        "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
        "Transport-moving", "Priv-house-serv", "Protective-serv",
        "Armed-Forces", "Others"
    ])
    
    relationship = st.sidebar.selectbox("Relationship", [
        "Wife", "Own-child", "Husband", "Not-in-family",
        "Other-relative", "Unmarried"
    ])
    
    race = st.sidebar.selectbox("Race", [
        "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"
    ])
    
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    
    # Financial Information
    st.sidebar.subheader("Financial Details")
    capital_gain = st.sidebar.number_input("Capital Gain", min_value=0, max_value=100000, value=0)
    capital_loss = st.sidebar.number_input("Capital Loss", min_value=0, max_value=5000, value=0)
    
    hours_per_week = st.sidebar.slider("Hours per Week", min_value=1, max_value=99, value=40)
    
    native_country = st.sidebar.selectbox("Native Country", [
        "United-States", "Cambodia", "England", "Puerto-Rico", "Canada",
        "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan",
        "Greece", "South", "China", "Cuba", "Iran", "Honduras",
        "Philippines", "Italy", "Poland", "Jamaica", "Vietnam",
        "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic",
        "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary",
        "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia",
        "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"
    ])
    
    return {
        'age': age,
        'workclass': workclass,
        'fnlwgt': fnlwgt,
        'educational-num': educational_num,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'gender': gender,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': native_country
    }

def encode_categorical_features(input_data):
    """Encode categorical features to match training data"""
    # Mapping dictionaries based on typical Adult dataset encoding
    encodings = {
        'workclass': {
            'Federal-gov': 0, 'Local-gov': 1, 'Others': 2, 'Private': 3,
            'Self-emp-inc': 4, 'Self-emp-not-inc': 5, 'State-gov': 6
        },
        'marital-status': {
            'Divorced': 0, 'Married-AF-spouse': 1, 'Married-civ-spouse': 2,
            'Married-spouse-absent': 3, 'Never-married': 4, 'Separated': 5, 'Widowed': 6
        },
        'occupation': {
            'Adm-clerical': 0, 'Armed-Forces': 1, 'Craft-repair': 2,
            'Exec-managerial': 3, 'Farming-fishing': 4, 'Handlers-cleaners': 5,
            'Machine-op-inspct': 6, 'Others': 7, 'Other-service': 8,
            'Priv-house-serv': 9, 'Prof-specialty': 10, 'Protective-serv': 11,
            'Sales': 12, 'Tech-support': 13, 'Transport-moving': 14
        },
        'relationship': {
            'Husband': 0, 'Not-in-family': 1, 'Other-relative': 2,
            'Own-child': 3, 'Unmarried': 4, 'Wife': 5
        },
        'race': {
            'Amer-Indian-Eskimo': 0, 'Asian-Pac-Islander': 1, 'Black': 2,
            'Other': 3, 'White': 4
        },
        'gender': {'Female': 0, 'Male': 1},
        'native-country': {
            'Cambodia': 0, 'Canada': 1, 'China': 2, 'Columbia': 3, 'Cuba': 4,
            'Dominican-Republic': 5, 'Ecuador': 6, 'El-Salvador': 7, 'England': 8,
            'France': 9, 'Germany': 10, 'Greece': 11, 'Guatemala': 12, 'Haiti': 13,
            'Holand-Netherlands': 14, 'Honduras': 15, 'Hong': 16, 'Hungary': 17,
            'India': 18, 'Iran': 19, 'Ireland': 20, 'Italy': 21, 'Jamaica': 22,
            'Japan': 23, 'Laos': 24, 'Mexico': 25, 'Nicaragua': 26,
            'Outlying-US(Guam-USVI-etc)': 27, 'Peru': 28, 'Philippines': 29,
            'Poland': 30, 'Portugal': 31, 'Puerto-Rico': 32, 'Scotland': 33,
            'South': 34, 'Taiwan': 35, 'Thailand': 36, 'Trinadad&Tobago': 37,
            'United-States': 38, 'Vietnam': 39, 'Yugoslavia': 40
        }
    }
    
    encoded_data = input_data.copy()
    for feature, mapping in encodings.items():
        if feature in encoded_data:
            encoded_data[feature] = mapping.get(encoded_data[feature], 0)
    
    return encoded_data

def main():
    # Header
    st.markdown('<h1 class="main-header">💼 Employee Salary Prediction</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model
    model, scaler, feature_names = load_model_and_scaler()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔮 Make Prediction")
        
        # Get input features
        input_data = create_input_features()
        
        # Display input data
        with st.expander("📋 Review Input Data", expanded=False):
            input_df = pd.DataFrame([input_data])
            st.dataframe(input_df, use_container_width=True)
        
        # Prediction button
        if st.button("🎯 Predict Salary", type="primary", use_container_width=True):
            # Encode categorical features
            encoded_data = encode_categorical_features(input_data)
            
            # Create feature array in correct order
            feature_array = np.array([[
                encoded_data['age'],
                encoded_data['workclass'],
                encoded_data['fnlwgt'],
                encoded_data['educational-num'],
                encoded_data['marital-status'],
                encoded_data['occupation'],
                encoded_data['relationship'],
                encoded_data['race'],
                encoded_data['gender'],
                encoded_data['capital-gain'],
                encoded_data['capital-loss'],
                encoded_data['hours-per-week'],
                encoded_data['native-country']
            ]])
            
            # Scale features
            scaled_features = scaler.transform(feature_array)
            
            # Make prediction
            prediction = model.predict(scaled_features)[0]
            probability = model.predict_proba(scaled_features)[0]
            
            # Display result
            if prediction == '>50K':
                st.markdown(f"""
                    <div class="prediction-box high-income">
                        <h2>💰 High Income Prediction</h2>
                        <h3>Predicted Salary: {prediction}</h3>
                        <p>Probability: {probability[1]:.2%}</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="prediction-box low-income">
                        <h2>💵 Lower Income Prediction</h2>
                        <h3>Predicted Salary: {prediction}</h3>
                        <p>Probability: {probability[0]:.2%}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Probability chart
            prob_df = pd.DataFrame({
                'Salary Class': ['≤50K', '>50K'],
                'Probability': probability
            })
            
            fig = px.bar(prob_df, x='Salary Class', y='Probability', 
                        title='Prediction Probabilities',
                        color='Salary Class',
                        color_discrete_map={'≤50K': '#ff6b6b', '>50K': '#4ecdc4'})
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.header("📊 Key Statistics")
        
        # Model info
        with st.expander("🤖 Model Information", expanded=True):
            try:
                with open('IBM/model_performance.txt', 'r') as f:
                    model_info = f.read()
                st.text(model_info)
            except FileNotFoundError:
                st.info("Model performance file not available")
        
        # Feature importance (for tree-based models)
        if hasattr(model, 'feature_importances_'):
            st.subheader("🎯 Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': ['age', 'workclass', 'fnlwgt', 'educational-num', 
                           'marital-status', 'occupation', 'relationship', 'race',
                           'gender', 'capital-gain', 'capital-loss', 'hours-per-week',
                           'native-country'],
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(importance_df.head(10), x='Importance', y='Feature', 
                        orientation='h', title='Top 10 Most Important Features')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Batch prediction section
    st.markdown("---")
    st.header("📂 Batch Prediction")
    
    uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=['csv'])
    
    if uploaded_file is not None:
        batch_data = pd.read_csv(uploaded_file)
        st.write(f"Uploaded data shape: {batch_data.shape}")
        st.write("Data preview:")
        st.dataframe(batch_data.head(), use_container_width=True)
        
        if st.button("🚀 Run Batch Prediction"):
            # Process batch data (simplified - assumes data is already preprocessed)
            try:
                # Scale the data
                scaled_batch = scaler.transform(batch_data)
                
                # Make predictions
                batch_predictions = model.predict(scaled_batch)
                batch_probabilities = model.predict_proba(scaled_batch)
                
                # Add predictions to dataframe
                batch_data['Predicted_Salary'] = batch_predictions
                batch_data['Probability_Low'] = batch_probabilities[:, 0]
                batch_data['Probability_High'] = batch_probabilities[:, 1]
                
                st.success("✅ Batch prediction completed!")
                st.dataframe(batch_data, use_container_width=True)
                
                # Download button
                csv = batch_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name='salary_predictions.csv',
                    mime='text/csv'
                )
                
            except Exception as e:
                st.error(f"Error in batch prediction: {str(e)}")
                st.info("Please ensure your CSV has the same features as the training data.")

if __name__ == "__main__":
    main()