import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, classification_report, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
def load_and_prepare_data():
    df = pd.read_csv('heart.csv')
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns

# Initialize models
def initialize_models():
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'K-Nearest Neighbor': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    return models

# Calculate evaluation metrics
def calculate_metrics(y_true, y_pred, y_pred_proba):
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_pred_proba),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    return metrics

# Train and evaluate all models
def train_and_evaluate_models():
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_prepare_data()
    
    print("Initializing models...")
    models = initialize_models()
    
    results = {}
    trained_models = {}
    
    print("Training and evaluating models...")
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Train the model
        if name == 'K-Nearest Neighbor':
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_pred_proba = model.decision_function(X_test)
                # Convert decision function to probabilities using sigmoid
                y_pred_proba = 1 / (1 + np.exp(-y_pred_proba))
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        results[name] = metrics
        trained_models[name] = model
        
        # Print results
        print(f"{name} Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        print()
    
    # Save models and scaler
    print("Saving models...")
    for name, model in trained_models.items():
        filename = name.lower().replace(' ', '_').replace('-', '_') + '.pkl'
        with open(f'model/{filename}', 'wb') as f:
            pickle.dump(model, f)
    
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature names
    with open('model/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names.tolist(), f)
    
    return results, y_test, trained_models, X_test, scaler

# Create comparison table
def create_comparison_table(results):
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df.round(4)
    return comparison_df

if __name__ == "__main__":
    # Create model directory if it doesn't exist
    import os
    if not os.path.exists('model'):
        os.makedirs('model')
    
    # Train and evaluate models
    results, y_test, trained_models, X_test, scaler = train_and_evaluate_models()
    
    # Create and display comparison table
    comparison_table = create_comparison_table(results)
    print("\n" + "="*80)
    print("MODEL COMPARISON TABLE")
    print("="*80)
    print(comparison_table)
    
    # Save results
    comparison_table.to_csv('model_results.csv')
    print("\nResults saved to 'model_results.csv'")
    print("Models saved to 'model/' directory")
