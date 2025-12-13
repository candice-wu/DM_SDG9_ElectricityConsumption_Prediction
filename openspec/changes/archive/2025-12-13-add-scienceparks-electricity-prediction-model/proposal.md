# Change: Add Science Parks Electricity Prediction Model

## Why
The project aims to implement an electricity prediction system for science parks in Taiwan to automatically predict and perform advanced data analysis, along with richer visualization work. This will provide an intuitive and interactive user experience through a Streamlit web application.

## What Changes
- Implementation of a comprehensive "Electricity Prediction system of science parks in Taiwan" as a 5-page Streamlit application.
- **Page 1: Data Exploration & Cleaning**: Feature for uploading, cleaning, and comparing raw vs. processed data.
- **Page 2: Electricity Consumption Prediction**: Interactive UI for predicting consumption using multiple selectable models (Linear Regression, Decision Tree, HistGradient, SVR, LightGBM) with hyperparameter tuning and a detailed feedback mechanism.
- **Page 3: Data Analysis**: In-depth analysis of feature correlations, linear regression, and categorical variable impacts using heatmaps, regression plots, and box plots.
- **Page 4: Data Transformation**: Demonstrates various data transformation techniques including normalization (Min-Max, Z-score), discretization (Binning, Decision Tree, Clustering), and data smoothing.
- **Page 5: Data Reduction**: A comprehensive module with features for:
  - **Dimensionality Reduction**: PCA, t-SNE, and feature ranking (Mutual Information, Information Gain, Distance Metrics like Euclidean and Hamming).
  - **Numerosity Reduction**: Parametric methods (Regression), non-parametric methods (Histograms, Clustering), and Sampling (Random, Stratified, Systematic).
  - **Data Compression**: DWT (Discrete Wavelet Transform) and PCA-based reconstruction.
- **Page 6: Model Comparison & Analysis**: Advanced dashboard for comparing multiple models via 'Prediction vs. Actual' plots, 'Residuals Plots', 'Feature Importance' analysis, and 'Confusion Matrix' for discretized results.
- Creation of a multi-page Streamlit web application for all deliverables.

## Impact
- Affected specs: New capability `scienceparks-electricity-prediction`.
- Affected code: New Python scripts for data processing, model training, visualization, and a 5-page Streamlit application.
