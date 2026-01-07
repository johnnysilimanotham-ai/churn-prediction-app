
# Bank Customer Churn Prediction Web Application
Project Overview

This project analyzes bank customer data to predict which clients are at risk of churn. The system is implemented as a Streamlit web application and allows users to upload datasets, clean and visualize data, train multiple machine learning models, compare performance, and deploy a model to make single or batch churn predictions.

Business Objective: Provide actionable insights that enable the bank to target retention strategies, reduce churn, and improve customer satisfaction.

Dataset

Source: Simulated bank customer dataset

Records: ~5,000 customers

Features: Age, Gender, Account Balance, Number of Products, Credit Score, Tenure, Geography, etc.

Preprocessing:

Missing values handled with mean, median, mode, or custom strategies

Target column normalized to 0 (no churn) / 1 (churn)

Unnecessary columns removed, duplicates dropped

System Setup

Platform & Environment:

Python 3.9+

Streamlit framework for web interface

Required Python Packages:

streamlit, numpy, pandas, scikit-learn, plotly, joblib

Application File:

streamlit_churn_app.py contains the full code for UI, preprocessing, modeling, evaluation, and predictions.

Run the App:

Primary Access (Live App): Streamlit Link

GitHub Repository: GitHub Link

Streamlit Cloud Instructions:

Log in to Streamlit Cloud
 with GitHub

Connect repository johnnysilimanotham-ai/beta2.0

Click “New App” → Streamlit Cloud installs dependencies and launches the app

Features & Functionality
1. Upload Data

Upload CSV file containing customer records

Preview dataset, check missing values, and select target column

Automatically normalizes target to 0/1

2. Data Cleaning

Drop unnecessary columns

Handle missing values (numeric/categorical) with multiple strategies

Remove duplicates

Store cleaned dataset for modeling

3. Data Visualization

Histograms and box plots for numeric features

Bar charts for categorical features

Missing data heatmaps

Correlation analysis

Automated recommendations for:

Imbalanced classes

High-cardinality features

Skewed distributions

4. Train Models

Train multiple models on cleaned data:

Logistic Regression

Random Forest

Support Vector Machine (SVM)

Preprocessing pipeline includes:

ColumnTransformer, SimpleImputer, StandardScaler, OneHotEncoder

Evaluate models using: Accuracy, Precision, Recall, F1-score, ROC-AUC

5. Model Comparison

Compare models side-by-side with charts and confusion matrices

Highlight best model based on F1-score

Deploy selected model on full dataset

6. Predictions

Single Customer: Input manually, see churn probability, risk level, and feature influence (for some models)

Batch Prediction: Upload CSV to generate churn probabilities and risk classifications; download results as CSV

Implemented Enhancements

Upload Tab Enhancements

Dataset preview panel for easy navigation

Improved Multicollinearity Analysis

Excludes target variable

Only shows feature-to-feature correlations

Corrected Model Retraining for Deployment

Ensures production model retrained on entire dataset

Batch Prediction via CSV Upload

Allows bulk churn predictions

Generates probability, prediction, and risk labels

Downloadable Prediction Results

Export CSV with feature values, churn probability, predicted label, and risk level

Skills & Tools Demonstrated

Programming & Tools: Python, Streamlit, GitHub

Data Science Techniques: Data Cleaning, Feature Engineering, Model Training (Logistic Regression, Random Forest, SVM), Model Evaluation, Batch & Single Predictions

Visualization: Matplotlib, Plotly

Project Skills: Problem-solving, translating business requirements into actionable insights, web application deployment
