# Employee Salary Prediction Project 💼

A machine learning project that predicts whether an employee's salary is above or below $50K based on demographic and work-related features.

## 🌟 Features

- **Interactive Web Application**: Built with Streamlit for easy user interaction
- **Multiple ML Models**: Compares RandomForest, Gradient Boosting, Logistic Regression, KNN, and SVM
- **Real-time Predictions**: Single employee prediction with probability scores
- **Batch Processing**: Upload CSV files for multiple predictions
- **Data Visualization**: Interactive charts showing prediction probabilities and feature importance
- **Model Performance**: Displays accuracy metrics and comparison charts

## 📊 Dataset

The project uses the Adult Income dataset (Census Income dataset) with the following features:

- **age**: Age of the individual
- **workclass**: Type of employment (Private, Self-employed, Government, etc.)
- **fnlwgt**: Final weight (demographic weighting)
- **education-num**: Number of years of education
- **marital-status**: Marital status
- **occupation**: Job category
- **relationship**: Family relationship
- **race**: Race/ethnicity
- **gender**: Gender
- **capital-gain**: Capital gains income
- **capital-loss**: Capital losses
- **hours-per-week**: Hours worked per week
- **native-country**: Country of origin
- **income**: Target variable (≤50K or >50K)

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the project files**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your dataset:**
   - Place your `adult.csv` file in the project directory
   - Ensure the CSV has the correct column names and format

### Usage

1. **Train the model:**
   ```bash
   python train_model.py
   ```
   This will:
   - Load and preprocess the data
   - Train multiple ML models
   - Select the best performing model
   - Save the trained model, scaler, and feature names

2. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

3. **Open your browser** and go to `http://localhost:8501`

## 📁 Project Structure

```
employee-salary-prediction/
│
├── train_model.py          # Model training script
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── adult.csv              # Dataset (you need to provide this)
│
└── Generated files after training:
    ├── best_model.pkl          # Trained model
    ├── scaler.pkl              # Feature scaler
    ├── feature_names.pkl       # Feature names
    ├── model_performance.txt   # Performance metrics
    └── model_comparison.png    # Model comparison chart
```

## 🔧 Data Preprocessing

The preprocessing pipeline includes:

1. **Missing Value Handling**: Replace '?' with 'Others'
2. **Outlier Removal**: Filter extreme age values and irrelevant categories
3. **Feature Engineering**: Remove redundant features (education text when education-num exists)
4. **Label Encoding**: Convert categorical variables to numerical
5. **Feature Scaling**: MinMax scaling for consistent feature ranges

## 🤖 Model Training

The system trains and compares multiple models:

- **Random Forest Classifier**
- **Gradient Boosting Classifier**
- **Logistic Regression**
- **K-Nearest Neighbors**
- **Support Vector Machine**

The best performing model (based on accuracy) is automatically selected and saved.

## 📱 Web Application Features

### Single Prediction
- Interactive sidebar with all input features
- Real-time prediction with probability scores
- Color-coded results (green for high income, red for low income)
- Interactive probability visualization

### Batch Prediction
- Upload CSV files for multiple predictions
- Automatic processing and prediction
- Download results with predictions and probabilities

### Analytics Dashboard
- Model performance metrics
- Feature importance visualization (for tree-based models)
- Interactive charts using Plotly

## 📈 Model Performance

Typical model performance on the Adult dataset:
- **Accuracy**: 85-87%
- **Precision**: High for both classes
- **Recall**: Balanced performance
- **F1-Score**: Good overall performance

## 🔍 Feature Importance

Key predictive features typically include:
1. **Age**: Older individuals tend to have higher incomes
2. **Education Level**: Higher education correlates with higher income
3. **Hours per Week**: More working hours often indicate higher income
4. **Occupation**: Managerial and professional roles earn more
5. **Marital Status**: Married individuals often have higher household income

## 🚨 Troubleshooting

### Common Issues

1. **"Model files not found" error:**
   - Run `python train_model.py` first to generate the model files

2. **CSV upload errors in batch prediction:**
   - Ensure your CSV has the same column structure as the training data
   - Check for missing or extra columns

3. **Low prediction accuracy:**
   - Verify data quality and preprocessing steps
   - Consider feature engineering or different algorithms

### Performance Tips

- For large datasets, consider using sample data for training
- Adjust model parameters in `train_model.py` for better performance
- Use GPU acceleration for SVM on large datasets

## 📝 Customization

### Adding New Features
1. Update the feature list in `create_input_features()` function
2. Modify the encoding logic in `encode_categorical_features()`
3. Retrain the model with new features

### Changing Models
1. Add new models to the `models` dictionary in `train_model.py`
2. Ensure compatibility with scikit-learn interface
3. Retrain and test performance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and test thoroughly
4. Submit a pull request with clear description

## 📄 License

This project is open-source and available under the MIT License.

## 🙏 Acknowledgments

- Adult Income dataset from UCI Machine Learning Repository
- Streamlit for the amazing web framework
- Scikit-learn for machine learning algorithms
- Plotly for interactive visualizations

---

**Happy Predicting! 🎯**
