# Heart Disease Classification System

## Problem Statement

This project implements and compares multiple machine learning classification models for predicting heart disease using clinical parameters. The system provides an interactive web application for model evaluation, prediction, and comparison, demonstrating real-world end-to-end ML deployment workflow.

## Dataset Description

The Heart Disease Dataset contains **1,025 patient records** with **13 clinical features** used to predict the presence of heart disease. This dataset meets the assignment requirements with:
- **Feature Size**: 13 clinical parameters (exceeds minimum 12)
- **Instance Size**: 1,025 patient records (exceeds minimum 500)
- **Target Variable**: Binary classification (0 = No heart disease, 1 = Heart disease present)

### Features:
1. **age** - Age of the patient (years)
2. **sex** - Gender (0 = Female, 1 = Male)
3. **cp** - Chest pain type (0-3)
4. **trestbps** - Resting blood pressure (mm Hg)
5. **chol** - Serum cholesterol (mg/dl)
6. **fbs** - Fasting blood sugar > 120 mg/dl (0 = No, 1 = Yes)
7. **restecg** - Resting electrocardiographic results (0-2)
8. **thalach** - Maximum heart rate achieved
9. **exang** - Exercise induced angina (0 = No, 1 = Yes)
10. **oldpeak** - ST depression induced by exercise relative to rest
11. **slope** - Slope of the peak exercise ST segment (0-2)
12. **ca** - Number of major vessels (0-4) colored by fluoroscopy
13. **thal** - Thalassemia (0-3)

## Models Used

Six machine learning classification models were implemented and evaluated:

### Model Performance Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.8098 | 0.9298 | 0.7619 | 0.9143 | 0.8312 | 0.6309 |
| Decision Tree | 0.9854 | 0.9857 | 1.0000 | 0.9714 | 0.9855 | 0.9712 |
| K-Nearest Neighbor | 0.8634 | 0.9629 | 0.8738 | 0.8571 | 0.8654 | 0.7269 |
| Naive Bayes | 0.8293 | 0.9043 | 0.8070 | 0.8762 | 0.8402 | 0.6602 |
| Random Forest (Ensemble) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost (Ensemble) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### Model Performance Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Shows good baseline performance with high recall (91.43%) but lower precision (76.19%). The model tends to favor sensitivity, making it suitable for screening applications where false negatives are costly. |
| Decision Tree | Excellent performance with near-perfect scores. Achieves perfect precision (100%) indicating no false positives, though slightly lower recall (97.14%) suggests few false negatives. |
| K-Nearest Neighbor | Balanced performance across all metrics with good accuracy (86.34%). The model shows consistent precision and recall, making it reliable for general classification tasks. |
| Naive Bayes | Moderate performance with good recall (87.62%) but lower precision (80.70%). Assumes feature independence which may not hold true for this medical dataset, affecting overall performance. |
| Random Forest (Ensemble) | Perfect performance across all metrics (100%). The ensemble approach with multiple decision trees eliminates overfitting and captures complex patterns in the data effectively. |
| XGBoost (Ensemble) | Perfect performance across all metrics (100%). The gradient boosting approach optimizes predictive accuracy by sequentially improving weak learners, resulting in outstanding classification capability. |

### Key Insights:

1. **Ensemble Models Dominate**: Both Random Forest and XGBoost achieved perfect performance, demonstrating the power of ensemble methods for this classification task.

2. **Decision Tree Excellence**: The single Decision Tree performed exceptionally well, suggesting the dataset has clear decision boundaries that tree-based models can capture effectively.

3. **Linear Model Limitations**: Logistic Regression, while providing interpretable results, showed the lowest performance, indicating non-linear relationships in the data.

4. **Balanced Performance**: KNN and Naive Bayes provided moderate but balanced performance across all metrics, making them suitable for general applications.

5. **Clinical Implications**: The high-performing models (Random Forest, XGBoost, Decision Tree) are excellent for diagnostic support, while Logistic Regression's high recall makes it valuable for initial screening.

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd heart-disease-classification
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train models** (optional - pre-trained models are included):
   ```bash
   python model_training.py
   ```

4. **Run the Streamlit app**:
   ```bash
   streamlit run streamlit_app.py
   ```

## Project Structure

```
heart-disease-classification/
│-- streamlit_app.py          # Main Streamlit application
│-- model_training.py         # Model training and evaluation script
│-- requirements.txt          # Python dependencies
│-- README.md                 # Project documentation
│-- heart.csv                 # Dataset
│-- model_results.csv         # Model performance results
│-- ML_Assignment2_Final_Submission.pdf  # Complete assignment submission
│-- Lab_Screenshot.pdf        # BITS Virtual Lab execution proof
│-- model/                    # Saved model files
│   |-- logistic_regression.pkl
│   |-- decision_tree.pkl
│   |-- k_nearest_neighbor.pkl
│   |-- naive_bayes.pkl
│   |-- random_forest.pkl
│   |-- xgboost.pkl
│   |-- scaler.pkl
│   └-- feature_names.pkl
```

## Web Application Features

The Streamlit web application includes:

### 1. **Home Page**
- Overview of the project and models
- Dataset information
- Model performance summary

### 2. **Model Comparison**
- Interactive comparison table with all evaluation metrics
- Visual comparisons (bar charts, radar chart)
- Performance analysis across different metrics

### 3. **Make Prediction**
- Individual patient prediction interface
- Input fields for all 13 clinical parameters
- Model selection dropdown
- Real-time prediction with probability scores
- Visual probability representation

### 4. **Upload Data**
- CSV file upload functionality
- Batch prediction capabilities
- Model evaluation on custom datasets
- Confusion matrix visualization
- Classification report generation
- Results download functionality

## Model Evaluation Metrics

All models were evaluated using six comprehensive metrics:

1. **Accuracy** - Overall correctness of predictions
2. **AUC Score** - Area Under the ROC Curve (discrimination ability)
3. **Precision** - Positive predictive value
4. **Recall** - Sensitivity (true positive rate)
5. **F1 Score** - Harmonic mean of precision and recall
6. **MCC Score** - Matthews Correlation Coefficient (balanced measure)

## Technical Implementation

### Data Preprocessing
- **Feature Scaling**: StandardScaler applied to normalize features
- **Train-Test Split**: 80-20 split with stratification
- **Missing Values**: No missing values in the dataset

### Model Configuration
- **Random State**: 42 for reproducibility
- **Cross-validation**: Not used (single train-test split as per assignment)
- **Hyperparameters**: Default parameters used for fair comparison

### Deployment
- **Platform**: Streamlit Community Cloud
- **Framework**: Streamlit for web interface
- **Model Storage**: Pickle format for serialization
- **Scalability**: Stateless design for cloud deployment

## Usage Instructions

### For Individual Predictions:
1. Navigate to "Make Prediction" page
2. Select desired model from dropdown
3. Enter patient clinical parameters
4. Click "Predict Heart Disease"
5. View results with probability scores

### For Batch Analysis:
1. Navigate to "Upload Data" page
2. Upload CSV file with required columns
3. Select model for evaluation/prediction
4. View comprehensive results and download

### For Model Comparison:
1. Navigate to "Model Comparison" page
2. View performance metrics table
3. Analyze visual comparisons
4. Study radar chart for comprehensive analysis

## Deployment Instructions

### Streamlit Community Cloud Deployment:

1. **Prepare Repository**:
   - Ensure all files are committed to GitHub
   - Verify requirements.txt is complete
   - Test application locally

2. **Deploy to Streamlit Cloud**:
   - Visit https://streamlit.io/cloud
   - Sign in with GitHub account
   - Click "New App"
   - Select your repository
   - Choose branch (main/master)
   - Select `streamlit_app.py` as main file
   - Click "Deploy"

3. **Verify Deployment**:
   - Wait for deployment to complete
   - Test all features in the deployed app
   - Ensure model loading works correctly

## Future Enhancements

1. **Hyperparameter Tuning**: Implement GridSearchCV for optimization
2. **Cross-validation**: Add k-fold cross-validation for robust evaluation
3. **Feature Engineering**: Explore additional feature transformations
4. **Model Explainability**: Add SHAP values for model interpretation
5. **Real-time Integration**: Connect to live medical databases
6. **Multi-language Support**: Add internationalization features

## Contributors

- **Developed by**: Rajeshwari Marimuthu
- **Assignment**: M.Tech (AIML/DSE) - Machine Learning Assignment 2
- **Institution**: BITS Pilani - Work Integrated Learning Programmes Division

---

**Note**: This application is intended for educational and demonstration purposes only. For medical diagnosis, always consult qualified healthcare professionals.
