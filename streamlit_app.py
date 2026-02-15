import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
import os

# Set page config
st.set_page_config(
    page_title="Heart Disease Classification",
    page_icon="❤️",
    layout="wide"
)

# Title and description
st.title("❤️ Heart Disease Classification App")
st.markdown("""
This application demonstrates multiple machine learning models for heart disease prediction.
Upload your CSV file to test the models or use the default dataset.
""")

# Load models and scaler
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Decision Tree': 'decision_tree.pkl',
        'K-Nearest Neighbor': 'k_nearest_neighbor.pkl',
        'Naive Bayes': 'naive_bayes.pkl',
        'Random Forest': 'random_forest.pkl',
        'XGBoost': 'xgboost.pkl'
    }
    
    for name, filename in model_files.items():
        try:
            with open(f'model/{filename}', 'rb') as f:
                models[name] = pickle.load(f)
        except FileNotFoundError:
            st.error(f"Model file {filename} not found!")
            return None
    
    # Load scaler
    with open('model/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Load feature names
    with open('model/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    return models, scaler, feature_names

# Load model results
@st.cache_data
def load_model_results():
    try:
        return pd.read_csv('model_results.csv', index_col=0)
    except FileNotFoundError:
        return None

# Main app
def main():
    # Load models and data
    models_data = load_models()
    if models_data is None:
        st.error("Could not load models. Please ensure model files are in the 'model' directory.")
        return
    
    models, scaler, feature_names = models_data
    model_results = load_model_results()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page:", ["Home", "Model Comparison", "Make Prediction", "Upload Data"])
    
    if page == "Home":
        st.header("Welcome to Heart Disease Classification System")
        
        st.markdown("""
        ### About this Application
        This application implements six different machine learning models for heart disease prediction:
        
        1. **Logistic Regression** - A linear model for binary classification
        2. **Decision Tree** - A tree-based model that makes decisions based on feature values
        3. **K-Nearest Neighbor** - A distance-based classification algorithm
        4. **Naive Bayes** - A probabilistic classifier based on Bayes' theorem
        5. **Random Forest** - An ensemble model using multiple decision trees
        6. **XGBoost** - A gradient boosting ensemble model
        
        ### Dataset Information
        - **Features**: 13 clinical parameters
        - **Target**: Heart disease presence (0 = No, 1 = Yes)
        - **Instances**: 1,025 patient records
        """)
        
        if model_results is not None:
            st.subheader("Model Performance Overview")
            st.dataframe(model_results.style.highlight_max(axis=0, color='lightgreen'))
    
    elif page == "Model Comparison":
        st.header("Model Performance Comparison")
        
        if model_results is not None:
            # Display comparison table
            st.subheader("Evaluation Metrics Comparison")
            st.dataframe(model_results.style.format("{:.4f}").highlight_max(axis=0, color='lightgreen'))
            
            # Create visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Accuracy Comparison")
                fig, ax = plt.subplots(figsize=(10, 6))
                model_results['Accuracy'].sort_values().plot(kind='barh', ax=ax, color='skyblue')
                ax.set_xlabel('Accuracy')
                ax.set_title('Model Accuracy Comparison')
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.subheader("AUC Score Comparison")
                fig, ax = plt.subplots(figsize=(10, 6))
                model_results['AUC'].sort_values().plot(kind='barh', ax=ax, color='lightcoral')
                ax.set_xlabel('AUC Score')
                ax.set_title('Model AUC Score Comparison')
                plt.tight_layout()
                st.pyplot(fig)
            
            # Metrics radar chart
            st.subheader("Comprehensive Metrics Comparison")
            metrics_to_plot = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']
            
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
            
            angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(model_results)))
            
            for i, (model_name, row) in enumerate(model_results.iterrows()):
                values = row[metrics_to_plot].tolist()
                values += values[:1]  # Complete the circle
                ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[i])
                ax.fill(angles, values, alpha=0.25, color=colors[i])
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics_to_plot)
            ax.set_ylim(0, 1)
            ax.set_title('Model Performance Radar Chart', size=16, pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.error("Model results not found!")
    
    elif page == "Make Prediction":
        st.header("Make Predictions")
        
        # Model selection
        selected_model = st.selectbox("Select a model:", list(models.keys()))
        
        st.subheader(f"Using {selected_model} for prediction")
        
        # Input fields for all features
        st.write("Enter patient data:")
        
        col1, col2, col3 = st.columns(3)
        
        input_data = {}
        
        with col1:
            input_data['age'] = st.number_input('Age', min_value=1, max_value=120, value=50)
            input_data['sex'] = st.selectbox('Sex (0=Female, 1=Male)', [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
            input_data['cp'] = st.selectbox('Chest Pain Type (0-3)', [0, 1, 2, 3])
            input_data['trestbps'] = st.number_input('Resting Blood Pressure', min_value=80, max_value=200, value=120)
            input_data['chol'] = st.number_input('Serum Cholesterol (mg/dl)', min_value=100, max_value=600, value=200)
        
        with col2:
            input_data['fbs'] = st.selectbox('Fasting Blood Sugar > 120 mg/dl (0=No, 1=Yes)', [0, 1])
            input_data['restecg'] = st.selectbox('Resting ECG Results (0-2)', [0, 1, 2])
            input_data['thalach'] = st.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=150)
            input_data['exang'] = st.selectbox('Exercise Induced Angina (0=No, 1=Yes)', [0, 1])
            input_data['oldpeak'] = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        
        with col3:
            input_data['slope'] = st.selectbox('Slope of Peak Exercise ST Segment (0-2)', [0, 1, 2])
            input_data['ca'] = st.selectbox('Number of Major Vessels (0-4)', [0, 1, 2, 3, 4])
            input_data['thal'] = st.selectbox('Thalassemia (0-3)', [0, 1, 2, 3])
        
        # Make prediction button
        if st.button("Predict Heart Disease"):
            # Prepare input data
            input_df = pd.DataFrame([input_data])
            
            # Ensure correct column order
            input_df = input_df[feature_names]
            
            # Scale the input
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            model = models[selected_model]
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0] if hasattr(model, 'predict_proba') else None
            
            # Display results
            st.subheader("Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("⚠️ **Heart Disease Detected**")
                    st.write("The model predicts that the patient has heart disease.")
                else:
                    st.success("✅ **No Heart Disease**")
                    st.write("The model predicts that the patient does not have heart disease.")
            
            with col2:
                if prediction_proba is not None:
                    st.write("Prediction Probabilities:")
                    st.write(f"- No Heart Disease: {prediction_proba[0]:.2%}")
                    st.write(f"- Heart Disease: {prediction_proba[1]:.2%}")
                    
                    # Create probability bar chart
                    fig, ax = plt.subplots(figsize=(8, 4))
                    labels = ['No Heart Disease', 'Heart Disease']
                    colors = ['green', 'red']
                    ax.bar(labels, prediction_proba, color=colors, alpha=0.7)
                    ax.set_ylabel('Probability')
                    ax.set_title('Prediction Probabilities')
                    ax.set_ylim(0, 1)
                    
                    # Add percentage labels on bars
                    for i, v in enumerate(prediction_proba):
                        ax.text(i, v + 0.01, f'{v:.1%}', ha='center', va='bottom')
                    
                    st.pyplot(fig)
    
    elif page == "Upload Data":
        st.header("Upload and Test Your Data")
        
        st.markdown("""
        Upload a CSV file with the following columns:
        age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
        """)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Load the uploaded data
                test_data = pd.read_csv(uploaded_file)
                st.success(f"File uploaded successfully! Found {len(test_data)} records.")
                
                # Display data preview
                st.subheader("Data Preview")
                st.dataframe(test_data.head())
                
                # Check if target column exists
                if 'target' in test_data.columns:
                    st.subheader("Model Evaluation on Uploaded Data")
                    
                    # Separate features and target
                    X_test = test_data.drop('target', axis=1)
                    y_test = test_data['target']
                    
                    # Ensure correct column order
                    X_test = X_test[feature_names]
                    
                    # Scale the data
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Model selection for evaluation
                    selected_model = st.selectbox("Select model for evaluation:", list(models.keys()), key="eval_model")
                    
                    if st.button("Evaluate Model"):
                        model = models[selected_model]
                        
                        # Make predictions
                        y_pred = model.predict(X_test_scaled)
                        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
                        
                        # Calculate metrics
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred)
                        recall = recall_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred)
                        
                        if y_pred_proba is not None:
                            auc = roc_auc_score(y_test, y_pred_proba)
                        else:
                            auc = "N/A"
                        
                        mcc = matthews_corrcoef(y_test, y_pred)
                        
                        # Display metrics
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Performance Metrics")
                            st.metric("Accuracy", f"{accuracy:.4f}")
                            st.metric("Precision", f"{precision:.4f}")
                            st.metric("Recall", f"{recall:.4f}")
                        
                        with col2:
                            st.metric("F1 Score", f"{f1:.4f}")
                            st.metric("AUC Score", f"{auc}" if auc != "N/A" else auc)
                            st.metric("MCC Score", f"{mcc:.4f}")
                        
                        # Confusion Matrix
                        st.subheader("Confusion Matrix")
                        cm = confusion_matrix(y_test, y_pred)
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        ax.set_title(f'Confusion Matrix - {selected_model}')
                        st.pyplot(fig)
                        
                        # Classification Report
                        st.subheader("Classification Report")
                        report = classification_report(y_test, y_pred, output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df.round(4))
                
                else:
                    st.subheader("Make Predictions on Uploaded Data")
                    st.info("No 'target' column found. Making predictions on all records...")
                    
                    # Prepare data
                    X_test = test_data[feature_names]
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Model selection
                    selected_model = st.selectbox("Select model for prediction:", list(models.keys()), key="pred_model")
                    
                    if st.button("Make Predictions"):
                        model = models[selected_model]
                        predictions = model.predict(X_test_scaled)
                        prediction_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
                        
                        # Add predictions to dataframe
                        result_df = test_data.copy()
                        result_df['prediction'] = predictions
                        
                        if prediction_proba is not None:
                            result_df['probability_no_disease'] = prediction_proba[:, 0]
                            result_df['probability_disease'] = prediction_proba[:, 1]
                        
                        # Display results
                        st.subheader("Prediction Results")
                        st.dataframe(result_df)
                        
                        # Download results
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
                        
                        # Summary statistics
                        st.subheader("Prediction Summary")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Total Predictions", len(predictions))
                            st.metric("Predicted Disease", sum(predictions))
                            st.metric("Predicted No Disease", len(predictions) - sum(predictions))
                        
                        with col2:
                            disease_rate = sum(predictions) / len(predictions) * 100
                            st.metric("Disease Prediction Rate", f"{disease_rate:.1f}%")
                            
                            if prediction_proba is not None:
                                avg_prob_disease = np.mean(prediction_proba[:, 1])
                                st.metric("Average Disease Probability", f"{avg_prob_disease:.1%}")
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.info("Please ensure your CSV file has the correct format and column names.")

if __name__ == "__main__":
    main()
