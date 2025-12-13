## Context
This project aims to predict electricity consumption in Taiwan's science parks using a provided dataset and various ML techniques, delivered via Streamlit. The system will involve complex data preprocessing, model selection, training, and robust visualization.

## Goals / Non-Goals
- Goals:
    - Develop an accurate electricity prediction model.
    - Provide advanced data analysis.
    - Create rich and interactive visualizations.
    - Deploy a user-friendly Streamlit web application.
- Non-Goals:
    - Real-time data ingestion from external APIs (at this stage).
    - Deployment to cloud other than Streamlit.

## Decisions
- Decision: Use CRISP-DM methodology for the project lifecycle.
- Alternatives considered: Agile, Waterfall (CRISP-DM is more suited for data science projects).

## Risks / Trade-offs
- Risk: Model accuracy might be limited by data quality/quantity. -> Mitigation: Extensive data preprocessing and exploration of various algorithms.
- Risk: Streamlit performance for complex visualizations. -> Mitigation: Optimize data loading and rendering, consider pre-rendering if necessary.

## Modeling Strategy
- **Initial Algorithm**: The project starts with a **Linear Regression** model, trained on-the-fly. This provides a solid baseline for prediction. 
- **Expanded Algorithms**: The system has been expanded to include `Decision Tree Regressor`, `HistGradientBoostingRegressor`, `Support Vector Regressor (SVR)`, and `LightGBM Regressor`.
- **Features**: The model is trained on `Year_EN`, `Month_NUM`, `Avg_Temperature`, and the custom-ordered, label-encoded location features (`Science_Park`, `Sub_Science_Park`, `County`, `Town`).
- **Caching**: All models and data are cached using `@st.cache_resource` and `@st.cache_data` to prevent retraining and reloading on every user interaction, improving performance.

## State Management
- **`st.session_state`**: The application heavily relies on Streamlit's session state to maintain a consistent user experience across pages and interactions.
- **Key State Variables**:
    - `data_loaded`: A boolean flag to indicate if a file has been uploaded and processed.
    - `df`, `cleaned_df`: The raw and cleaned pandas DataFrames.
    - `preprocessor`: The instance of the `DataPreprocessor` class.
    - `selected_park`, `selected_sub_park`, `selected_year`, `selected_month`: User selections from the sidebar.
    - `avg_temp`: The temperature value used for prediction, either from historical data or user input.
    - `prediction_result`: A dictionary holding the predicted and actual usage values to persist them across reruns for the feedback loop.
    - Various keys for storing results from the 'Data Reduction' page analyses to ensure persistence.

## Streamlit UI Design
The application is structured as a multi-page app to separate concerns, providing a clean and intuitive user workflow. The pages are organized as follows:

### Main Page: 📄 資料探索與清理 (Data Exploration & Cleaning) (`5114050013_DM_SDG9.py`)
- **Purpose**: To upload, process, and inspect the data. This is the entry point of the application.
- **Layout**: Wide layout (`st.set_page_config(layout="wide")`) with two parallel columns for side-by-side comparison.
- **Components**:
    1.  **File Uploader**: Allows the user to upload the raw CSV data.
    2.  **Raw Data Display**: Shows the full raw DataFrame with detailed info (shape, data types, styled missing values) in an expander.
    3.  **Cleaned Data Display**: Shows the full cleaned DataFrame (after processing via `DataPreprocessor`) with detailed info in an expander.
- **Workflow**: Upon successful file upload, the raw (`df`), cleaned (`cleaned_df`) DataFrames, and the `DataPreprocessor` object are stored in `st.session_state`, making them available to all other pages.

### Page 1: 🔋 用電量預測 (Electricity Consumption Prediction) (`pages/1_🔋_用電量預測.py`)
- **Purpose**: To configure prediction parameters, run forecasts using various models, receive feedback, and evaluate model performance.
- **Workflow**: This page first checks `st.session_state` for the cleaned data. If not found, it prompts the user to return to the main page to upload a file.
- **Sidebar Components**:
    - **Model Selection**: Dropdown to select the prediction model. Options include "線性迴歸", "決策樹迴歸", "梯度提升樹迴歸 (HistGradient)", "SVR", and "梯度提升樹迴歸 (LightGBM)".
    - **Hyperparameter Tuning**: Dynamic sliders and number inputs appear based on the selected model to tune its hyperparameters.
    - **Location Selection**: Cascading, stateful dropdowns for "科學園區" and "子園區" with custom North/Central/South sorting.
    - **Date Selection**: A text input for "西元年" (validated for 4 digits) and a dropdown for "月份".
    - **Temperature Input ("Smart Default")**: A number input for "月均溫(°C)" which is auto-filled with historical data but is user-overridable.
    - **Action Buttons**: "💡 預估 (Predict)" and "🧹 清除 (Clear)" buttons.
- **Main Panel Components**:
    - **Prediction Result**: Displays the "用電量預估值 (萬KW)" using `st.markdown` with custom styling.
    - **Feedback Loop**: Provides an optional input for actual usage, calculates and displays the percentage difference with a Plotly gauge chart, conditional emojis, and messages, triggering `st.balloons()` for exact matches.
    - **Model Performance Evaluation**: Displays key metrics (`R-squared`, `RMSE`, `MAE`) in columns using `st.metric`.

### Page 2: 📊 資料分析 (Data Analysis) (`pages/2_📊_資料分析.py`)
- **Purpose**: To visualize relationships and patterns within the cleaned data.
- **Workflow**: Requires cleaned data to be present in `st.session_state`.
- **Sections**:
    1.  **特徵相關性分析 (Correlation Analysis)**: An annotated heatmap displaying the correlation matrix of all numerical features.
    2.  **數值型變數與用電量關係 (Numerical Variables & Electricity Usage)**: A linear regression plot (`seaborn.jointplot`) and a residual plot to analyze the relationship between temperature and electricity usage.
    3.  **類別型變數與用電量關係 (Categorical Variables & Electricity Usage)**: A series of box plots showing the distribution of electricity usage across different categorical features (parks, counties, etc.).
    4.  **關聯規則分析 (Association Rule Mining)**:
        - **Purpose**: To discover "if-then" relationships and frequent patterns between selected attributes in the dataset, after appropriate discretization of continuous variables.
        - **UI Components**:
            - **Feature Selector**: `st.multiselect` to allow users to select single or multiple categorical features (and continuous features which will be discretized).
            - **Discretization Settings**: For selected continuous features (e.g., `Avg_Temperature`, `Electricity_Usage`), provide input fields (e.g., `st.number_input`) to specify the number of bins for discretization (e.g., using equal-depth binning).
            - **Rule Parameters**: `st.slider` widgets for setting `Min Support` and `Min Confidence` thresholds.
            - **Execution Button**: A `st.button` to trigger the Apriori algorithm and association rule generation.
        - **Results Display**:
            - Display the generated association rules as an `st.dataframe`, including `Antecedents`, `Consequents`, `Support`, `Confidence`, and `Lift` metrics.
            - Provide an `st.expander` with explanatory text on how to interpret the results, especially the `Lift` value (Lift > 1 implies positive correlation, < 1 implies negative correlation, = 1 implies independence).
    - **Dynamic Conclusions**: Each plot is accompanied by an expandable conclusion section that provides a static "Method Definition" and a dynamic, data-driven "Data Characteristics" analysis.

### Page 3: ♻️ 資料轉換 (Data Transformation) (`pages/3_♻️_資料轉換.py`)
- **Purpose**: To demonstrate the effects of various data transformation techniques.
- **Workflow**: Fetches original and transformed data from the `DataPreprocessor` object.
- **Sections**:
    1.  **原始資料概覽 (Raw Data Overview)**: Compares the distributions and statistics of the original 'Electricity Consumption' and 'Average Temperature' features.
    2.  **正規化 (Normalization)**: Visualizes and explains Min-Max, Z-score, and Decimal Scaling normalization methods for the key numerical features.
    3.  **離散化 (Discretization)**: An interactive section allowing users to apply and compare Equal-width Binning, Equal-depth Binning, Decision Tree analysis, Clustering analysis, and Chi-Squared (χ²) analysis.
    4.  **資料平滑化 (Data Smoothing)**: An interactive section to demonstrate smoothing by bin means, medians, and boundaries.

### Page 4: 🔬 資料精簡 (Data Reduction) (`pages/4_🔬_資料精簡.py`)
- **Purpose**: To explore and apply various data reduction techniques.
- **UI Structure**: A three-tab layout: "維度縮減 (Dimensionality Reduction)", "數量縮減 (Numerosity Reduction)", and "資料壓縮 (Data Compression)".
- **Features**:
    - **Dimensionality Reduction**:
        - **Visual Reduction**: Interactive PCA and t-SNE plots for visualizing high-dimensional data.
        - **Feature Ranking & Distance Metrics**: Tools to rank features using Mutual Information and Information Gain, and to calculate distances between data points (Euclidean, Manhattan, Chebyshev, Minkowski, Hamming).
    - **Numerosity Reduction**:
        - **Parametric Methods**: Demonstrates using Linear and Decision Tree regression models to summarize data.
        - **Non-parametric Methods**: Includes interactive demos for Histograms, Clustering (K-Means), and Sampling (Random, Stratified, Systematic).
    - **Data Compression**:
        - **DWT (Discrete Wavelet Transform)**: Compresses a signal and visualizes the comparison between the original and reconstructed signal.
        - **PCA Compression**: Uses PCA to compress and reconstruct data, showing the trade-off between compression and reconstruction error (MSE).

### Page 5: 🎭 模型比較與進階分析 (Model Comparison & Analysis) (`pages/5_🎭_模型比較分析.py`)
- **Purpose**: An interactive dashboard for in-depth comparison of the trained regression models.
- **Components**:
    1.  **Multi-Model Selector**: Allows users to select one or more models for comparison.
    2.  **Prediction vs. Actual Plot**: A scatter plot comparing the predicted values against the actual values for the selected models.
    3.  **Residuals Plot**: A scatter plot of residuals to diagnose model bias and variance.
    4.  **Single Model Analysis**: When a single model is selected, two new sections appear:
        - **Feature Importance**: An Altair bar chart showing the most influential features for the model.
        - **Confusion Matrix Analysis**: A Plotly heatmap showing the model's accuracy on discretized prediction buckets, providing insight into how the model performs at different usage levels.

## Migration Plan
N/A (New project)
