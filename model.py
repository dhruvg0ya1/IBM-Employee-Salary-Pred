import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(csv_path):
    """Load and preprocess the Adult dataset"""
    print("Loading data...")
    data = pd.read_csv(csv_path)
    print(f"Original data shape: {data.shape}")
    
    # Handle missing values (? symbols)
    print("Handling missing values...")
    data['workclass'].replace({'?': 'Others'}, inplace=True)
    data['occupation'].replace({'?': 'Others'}, inplace=True)
    data['native-country'].replace({'?': 'Others'}, inplace=True)
    
    # Remove outliers and irrelevant categories
    print("Removing outliers and cleaning data...")
    # Age outliers
    data = data[(data['age'] <= 75) & (data['age'] >= 17)]
    
    # Remove categories with very low counts that don't contribute to income prediction
    data = data[data['workclass'] != 'Without-pay']
    data = data[data['workclass'] != 'Never-worked']
    
    # Remove very low education categories
    data = data[~data['education'].isin(['1st-4th', '5th-6th', 'Preschool'])]
    
    print(f"Data shape after cleaning: {data.shape}")
    
    # Drop redundant education column (education-num provides same info)
    data.drop(columns=['education'], inplace=True)
    
    # Encode categorical variables
    print("Encoding categorical variables...")
    encoder = LabelEncoder()
    categorical_columns = ['workclass', 'marital-status', 'occupation', 
                          'relationship', 'race', 'gender', 'native-country']
    
    for col in categorical_columns:
        data[col] = encoder.fit_transform(data[col])
    
    return data

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and return the best one"""
    print("Training multiple models...")
    
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel='rbf', random_state=42)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        trained_models[name] = model
        
        print(f"{name} Accuracy: {accuracy:.4f}")
        print(f"Classification Report for {name}:")
        print(classification_report(y_test, y_pred))
        print("-" * 50)
    
    # Find best model
    best_model_name = max(results, key=results.get)
    best_model = trained_models[best_model_name]
    
    print(f"\n✅ Best model: {best_model_name} with accuracy: {results[best_model_name]:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), results.values(), color='skyblue')
    plt.title('Model Comparison - Accuracy Scores')
    plt.ylabel('Accuracy Score')
    plt.xlabel('Models')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return best_model, best_model_name, results

def main():
    # Load and preprocess data
    data = load_and_preprocess_data('C:\\Users\\Dhruv\\OneDrive\\Documents\\Python\\IBM\\adult.csv')
    
    # Split features and target
    X = data.drop(columns=['income'])
    y = data['income']
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # Scale features
    print("Scaling features...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Train models
    best_model, best_model_name, results = train_models(X_train, X_test, y_train, y_test)
    
    # Save the best model and scaler
    joblib.dump(best_model, 'IBM/best_model.pkl')
    joblib.dump(scaler, 'IBM/scaler.pkl')

    # Save feature names for the app
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, 'IBM/feature_names.pkl')

    print(f"\n✅ Best model ({best_model_name}) saved as 'best_model.pkl'")
    print("✅ Scaler saved as 'scaler.pkl'")
    print("✅ Feature names saved as 'feature_names.pkl'")
    
    # Save model performance report
    with open('IBM/model_performance.txt', 'w') as f:
        f.write("Employee Salary Prediction - Model Performance Report\n")
        f.write("=" * 60 + "\n\n")
        for name, accuracy in results.items():
            f.write(f"{name}: {accuracy:.4f}\n")
        f.write(f"\nBest Model: {best_model_name} (Accuracy: {results[best_model_name]:.4f})\n")
    
    print("✅ Performance report saved as 'model_performance.txt'")

if __name__ == "__main__":
    main()