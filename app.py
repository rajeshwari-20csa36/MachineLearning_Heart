import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Page configuration
st.set_page_config(
    page_title="ML Classification Models",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 8px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🤖 Machine Learning Classification Models Comparison")
st.markdown("### M.Tech (AIML/DSE) - Assignment 2: Heart Disease Classification")
st.markdown("---")

# Model mapping with proper names
model_mapping = {
    'Logistic Regression': 'model_logistic_regression.pkl',
    'Decision Tree': 'model_decision_tree.pkl',
    'K-Nearest Neighbors': 'model_knearest_neighbors.pkl',
    'Naive Bayes': 'model_naive_bayes.pkl',
    'Random Forest': 'model_random_forest.pkl',
    'XGBoost': 'model_xgboost.pkl'
}

# Sidebar for model selection and data upload
with st.sidebar:
    st.header("📊 Configuration Panel")
    st.markdown("---")
    
    # Model Selection
    st.subheader("1. Select Model")
    selected_model_name = st.selectbox(
        "Choose a classification model:",
        list(model_mapping.keys()),
        help="Select one of the 6 implemented models"
    )
    
    st.markdown("---")
    
    # Data Upload
    st.subheader("2. Upload Test Data")
    st.info("⚠️ **Note:** Upload test data in CSV format with a 'target' column")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="CSV must include 'target' column for predictions"
    )
    
    st.markdown("---")
    
    # Instructions
    st.subheader("📋 Instructions")
    with st.expander("How to use this app"):
        st.markdown("""
        1. **Select a Model** from the dropdown
        2. **Upload Test Data** (CSV with 'target' column)
        3. **View Results** including:
           - Evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
           - Confusion Matrix
           - Classification Report
           - Feature Importance (for tree-based models)
        """)

# Main content area
if uploaded_file is not None:
    try:
        # Load dataset
        df = pd.read_csv(uploaded_file)
        
        # Display success message and dataset info
        st.success(f"✅ Dataset loaded successfully!")
        
        # Dataset Preview Section
        with st.expander("📊 Dataset Preview", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", df.shape[0])
            with col2:
                st.metric("Total Columns", df.shape[1])
            with col3:
                if 'target' in df.columns:
                    st.metric("Classes", df['target'].nunique())
            
            st.dataframe(df.head(10), use_container_width=True)
            
            # Basic statistics
            st.subheader("Dataset Statistics")
            st.dataframe(df.describe(), use_container_width=True)
        
        # Validate target column
        if 'target' not in df.columns:
            st.error("❌ Error: Dataset must contain a 'target' column!")
            st.stop()
        
        # Prepare data
        X_test = df.drop('target', axis=1)
        y_test = df['target']
        
        # Load scaler for scaling-dependent models
        scaler = None
        if os.path.exists('scaler.pkl'):
            try:
                with open('scaler.pkl', 'rb') as f:
                    scaler = pickle.load(f)
            except:
                st.warning("⚠️ Scaler not found, using raw features for all models")
        
        # Determine if scaling is needed
        needs_scaling = selected_model_name in ['Logistic Regression', 'K-Nearest Neighbors']
        
        if needs_scaling and scaler is not None:
            X_test_model = scaler.transform(X_test)
        else:
            X_test_model = X_test.values
        
        # Load model
        model_file = model_mapping[selected_model_name]
        
        if not os.path.exists(model_file):
            st.error(f"❌ Model file '{model_file}' not found!")
            st.info("Please ensure all model files are trained and saved.")
            st.stop()
        
        try:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            st.stop()
        
        st.success(f"✅ {selected_model_name} model loaded successfully!")
        
        # Make predictions
        try:
            y_pred = model.predict(X_test_model)
            y_pred_proba = model.predict_proba(X_test_model)
        except Exception as e:
            st.error(f"❌ Error making predictions: {str(e)}")
            st.stop()
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Handle AUC calculation for binary/multi-class
        n_classes = len(np.unique(y_test))
        try:
            if n_classes == 2:
                auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
        
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        # Display metrics in organized layout
        st.markdown("---")
        st.subheader(f"📈 {selected_model_name} - Evaluation Metrics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{accuracy:.4f}")
            st.metric("Precision", f"{precision:.4f}")
        with col2:
            st.metric("AUC Score", f"{auc:.4f}")
            st.metric("Recall", f"{recall:.4f}")
        with col3:
            st.metric("F1 Score", f"{f1:.4f}")
            st.metric("MCC Score", f"{mcc:.4f}")
        
        # Store metrics for comparison table
        metrics_data = {
            'Model': selected_model_name,
            'Accuracy': f"{accuracy:.4f}",
            'AUC': f"{auc:.4f}",
            'Precision': f"{precision:.4f}",
            'Recall': f"{recall:.4f}",
            'F1': f"{f1:.4f}",
            'MCC': f"{mcc:.4f}"
        }
        
        # Display metrics table
        st.markdown("---")
        st.subheader("📋 Metrics Summary")
        metrics_df = pd.DataFrame([metrics_data])
        st.dataframe(metrics_df, use_container_width=True)
        
        # Confusion Matrix
        st.markdown("---")
        st.subheader("🎯 Confusion Matrix")
        
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=np.unique(y_test),
                   yticklabels=np.unique(y_test),
                   cbar_kws={'label': 'Count'})
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        ax.set_title(f'Confusion Matrix - {selected_model_name}')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Classification Report
        st.markdown("---")
        st.subheader("📊 Classification Report")
        
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        
        # Highlight the report
        st.dataframe(
            report_df.style.format('{:.4f}'),
            use_container_width=True
        )
        
        # Feature Importance (for tree-based models)
        if selected_model_name in ['Decision Tree', 'Random Forest', 'XGBoost']:
            st.markdown("---")
            st.subheader("🌲 Feature Importance")
            
            try:
                feature_importance = pd.DataFrame({
                    'Feature': X_test.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                top_n = min(10, len(feature_importance))
                sns.barplot(data=feature_importance.head(top_n), 
                           x='Importance', y='Feature', 
                           palette='viridis')
                ax.set_title(f'Top {top_n} Important Features - {selected_model_name}')
                ax.set_xlabel('Importance Score')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Display full feature importance table
                with st.expander("📈 View All Feature Importances"):
                    st.dataframe(
                        feature_importance.style.format({'Importance': '{:.6f}'}),
                        use_container_width=True
                    )
            except Exception as e:
                st.warning(f"Could not display feature importance: {str(e)}")
        
        # Prediction Statistics
        st.markdown("---")
        st.subheader("📊 Prediction Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Predictions", len(y_pred))
            st.metric("Correct Predictions", (y_pred == y_test).sum())
        with col2:
            st.metric("Incorrect Predictions", (y_pred != y_test).sum())
            st.metric("Error Rate", f"{(1 - accuracy) * 100:.2f}%")
        
    except pd.errors.EmptyDataError:
        st.error("❌ Error: The uploaded file is empty!")
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")

else:
    # Welcome section
    st.info("👈 Upload a test dataset (CSV format) from the sidebar to begin model evaluation")
    
    st.markdown("### 🎯 Quick Start Guide:")
    st.markdown("""
    #### Step 1: Prepare Your Data
    - Download or prepare a CSV file with features and a 'target' column
    - Ensure column names match the training dataset
    
    #### Step 2: Select a Model
    - Choose from 6 implemented models via the sidebar dropdown
    
    #### Step 3: Upload & Evaluate
    - Upload your CSV file using the file uploader
    - View comprehensive evaluation metrics including:
        - **Accuracy, AUC, Precision, Recall, F1 Score, MCC**
        - **Confusion Matrix** for error analysis
        - **Classification Report** with per-class metrics
        - **Feature Importance** (for tree-based models)
    """)
    
    st.markdown("---")
    st.markdown("### 📋 Supported Models:")
    models_info = {
        '1️⃣ Logistic Regression': 'Linear classifier for binary/multi-class problems',
        '2️⃣ Decision Tree': 'Tree-based classifier with interpretable rules',
        '3️⃣ K-Nearest Neighbors': 'Instance-based classifier using similarity',
        '4️⃣ Naive Bayes': 'Probabilistic classifier based on Bayes theorem',
        '5️⃣ Random Forest': 'Ensemble of decision trees for robust predictions',
        '6️⃣ XGBoost': 'Gradient boosting for high-performance classification'
    }
    
    for model, desc in models_info.items():
        st.markdown(f"- **{model}**: {desc}")
    
    st.markdown("---")
    st.markdown("### 📊 Evaluation Metrics Explained:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - **Accuracy**: Proportion of correct predictions
        - **Precision**: Accuracy of positive predictions
        - **Recall**: Ability to find all positive instances
        """)
    with col2:
        st.markdown("""
        - **F1 Score**: Harmonic mean of Precision & Recall
        - **AUC Score**: Area under ROC curve (0.5-1.0)
        - **MCC**: Correlation coefficient (-1 to 1)
        """)
