import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.data_preprocessing import DataPreprocessor # Import DataPreprocessor
from src.ui_components import render_app_info
import plotly.graph_objects as go # Added for gauge chart

# --- Page Configuration ---
st.set_page_config(page_title="用電量預測", page_icon="🔋", layout="wide")

# --- Model Name Mapping ---
MODEL_NAME_MAPPING = {
    "線性迴歸 (Linear Regression)": "LinearRegression",
    "決策樹迴歸 (Decision Tree Regression)": "DecisionTreeRegressor",
    "梯度提升樹迴歸 (HistGradient)": "HistGradientBoostingRegressor",
    "支持向量迴歸 (SVR)": "SVR",
    "梯度提升樹迴歸 (LightGBM)": "LGBMRegressor"
}

# --- UI & Helper Functions ---

def create_gauge_chart(percentage_error):
    """Creates a Plotly gauge chart for the prediction error."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percentage_error,
        title={'text': "誤差程度", 'font': {'size': 20}},
        number={'suffix': "%", 'font': {'size': 28}, 'valueformat': '.2f'},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#2E3B4E"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 3], 'color': '#77DD77'},  # Green
                {'range': [3, 5], 'color': '#FFDD77'},  # Yellow
                {'range': [5, 100], 'color': '#FF6961'}   # Red
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 5
            }
        }
    ))
    fig.update_layout(
        height=250, 
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig

# Initialize core keys if they don't exist
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = None
if 'trained_model' not in st.session_state: # To store the actual trained model for future use if needed
    st.session_state.trained_model = None

def get_sub_park_options(park_name):
    """Helper to get sorted sub-park options for a given park."""
    if not park_name or 'df' not in st.session_state:
        return []
    original_df = st.session_state.get('df', pd.DataFrame())
    preprocessor = st.session_state.get('preprocessor')
    
    sub_park_names_all = sorted(original_df[original_df['Science_Park'] == park_name]['Sub_Science_Park'].unique())
    
    if preprocessor and 'Sub_Science_Park' in preprocessor.encoders:
        sub_park_custom_order = preprocessor.encoders['Sub_Science_Park']
        # Filter and reorder based on custom order
        return [name for name in sub_park_custom_order if name in sub_park_names_all]
    return sub_park_names_all

def _calculate_smart_default_avg_temp():
    """Calculates smart default average temperature based on selected sub-park, year, and month."""
    if not all(key in st.session_state for key in ['cleaned_df', 'df', 'selected_sub_park', 'selected_year', 'selected_month']) or not st.session_state.selected_sub_park:
        return 25.0 # Fallback default

    cleaned_df = st.session_state['cleaned_df']
    original_df = st.session_state['df']
    
    # Map selected sub-park name back to its encoded code for lookup in cleaned_df
    sub_park_name_to_code_map = {}
    if 'preprocessor' in st.session_state and 'Sub_Science_Park' in st.session_state['preprocessor'].encoders:
        for idx, name in enumerate(st.session_state['preprocessor'].encoders['Sub_Science_Park']):
            sub_park_name_to_code_map[name] = idx

    default_temp = 25.0 # Initial fallback

    try:
        selected_year_int = int(st.session_state.selected_year)
        selected_sub_park_name = st.session_state.selected_sub_park
        selected_month_num = st.session_state.selected_month
        
        if selected_sub_park_name and selected_sub_park_name in sub_park_name_to_code_map:
            selected_sub_park_code = sub_park_name_to_code_map[selected_sub_park_name]
            
            # First, try to get actual temperature from the same year, month, sub_park
            historical_data_exact = cleaned_df[
                (cleaned_df['Sub_Science_Park'] == selected_sub_park_code) &
                (cleaned_df['Year_EN'] == selected_year_int) &
                (cleaned_df['Month_NUM'] == selected_month_num)
            ]
            if not historical_data_exact.empty and not pd.isna(historical_data_exact['Avg_Temperature'].iloc[0]):
                default_temp = historical_data_exact['Avg_Temperature'].iloc[0]
            else:
                # If not found for exact year, get average from all available years for that sub_park and month
                hist_avg_df = cleaned_df[
                    (cleaned_df['Sub_Science_Park'] == selected_sub_park_code) &
                    (cleaned_df['Month_NUM'] == selected_month_num)
                ].dropna(subset=['Avg_Temperature'])
                if not hist_avg_df.empty:
                    default_temp = hist_avg_df['Avg_Temperature'].mean()
    except (ValueError, TypeError, IndexError, KeyError):
        pass # Keep default_temp as fallback
    
    return float(f"{default_temp:.2f}")

def clear_results_and_update_temp():
    """Clears prediction results and updates the smart default temperature."""
    st.session_state.prediction_result = None
    st.session_state.model_metrics = None
    st.session_state.avg_temp = _calculate_smart_default_avg_temp()

def on_park_change():
    """Callback for when the main science park is changed."""
    new_park = st.session_state.selected_park
    sub_park_options = get_sub_park_options(new_park)
    if sub_park_options:
        st.session_state.selected_sub_park = sub_park_options[0]
    else:
        st.session_state.selected_sub_park = None
    clear_results_and_update_temp()

def initialize_state(original_df):
    """
    Initializes all necessary session state keys for the sidebar widgets
    if they don't exist. This should only run once per session.
    """
    if 'state_initialized' in st.session_state:
        return

    park_order = ["新竹科學園區", "中部科學園區", "南部科學園區"]
    available_parks = [park for park in park_order if park in original_df['Science_Park'].unique()]

    # Set defaults only if keys are missing
    if 'selected_park' not in st.session_state and available_parks:
        st.session_state.selected_park = available_parks[0]
    
    if 'selected_sub_park' not in st.session_state and st.session_state.get('selected_park'):
        sub_park_options = get_sub_park_options(st.session_state.selected_park)
        if sub_park_options:
            st.session_state.selected_sub_park = sub_park_options[0]

    if 'selected_year' not in st.session_state:
        st.session_state.selected_year = str(datetime.datetime.now().year)
    
    if 'selected_month' not in st.session_state:
        st.session_state.selected_month = datetime.datetime.now().month

    if 'avg_temp' not in st.session_state:
        st.session_state.avg_temp = _calculate_smart_default_avg_temp()
    
    if 'model_selection' not in st.session_state:
        st.session_state.model_selection = "線性迴歸 (Linear Regression)" # Default model

    # Initialize hyperparameter defaults directly with correct types
    if 'dt_max_depth' not in st.session_state: st.session_state.dt_max_depth = 5
    if 'hgb_learning_rate' not in st.session_state: st.session_state.hgb_learning_rate = 0.1
    if 'hgb_max_iter' not in st.session_state: st.session_state.hgb_max_iter = 100
    if 'svr_c' not in st.session_state: st.session_state.svr_c = 100.0 # Ensure float
    if 'svr_gamma' not in st.session_state: st.session_state.svr_gamma = 0.1 # Ensure float
    if 'lgbm_n_estimators' not in st.session_state: st.session_state.lgbm_n_estimators = 100
    if 'lgbm_learning_rate' not in st.session_state: st.session_state.lgbm_learning_rate = 0.1 # Ensure float

    st.session_state.state_initialized = True

def reset_prediction_form(original_df):
    """Resets the form to its initial default state."""
    keys_to_reset = [
        'prediction_result', 'actual_usage_input', 'state_initialized',
        'selected_park', 'selected_sub_park', 'selected_year',
        'selected_month', 'avg_temp', 'model_selection', 'model_metrics',
        'trained_model', 'model_scaler', # Also reset the scaler
        # Hyperparameter keys to reset
        'dt_max_depth', 'hgb_learning_rate', 'hgb_max_iter', 'svr_c', 'svr_gamma', 'lgbm_n_estimators', 'lgbm_learning_rate'
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    # Re-initialize to get fresh defaults
    initialize_state(original_df)

# --- Main App ---
st.title("🔋 用電量預測 (Electricity Consumption Prediction)")

# Render the static info sections in the sidebar
render_app_info()

st.info("""
        此頁面讓您設定參數，可對特定科學園區的未來用電量進行預測

            ⚠️ 已知限制：目前因 Streamlit 的頁面重載機制，切換分頁後，此側邊欄的條件值會回復至系統預設值。
               如有需要，請重新挑選選單或設定條件值並重新預估用電量，造成不便請見諒，本系統將持續改進！
        """,icon="ℹ️")

if not st.session_state.get('data_loaded'):
    st.warning("⬅️ 請先至「📄 資料探索與清理」頁面上傳並清理資料")
else:
    cleaned_df = st.session_state['cleaned_df']
    original_df = st.session_state['df']
    preprocessor = st.session_state['preprocessor']
    
    # Initialize all states at the beginning, passing the dataframe
    initialize_state(original_df)

    # Prepare data for all models once (unscaled X, y)
    X_full, y_full, features, target = preprocessor.get_prediction_data(cleaned_df)
    
    park_name_to_code = dict(zip(original_df['Science_Park'], cleaned_df['Science_Park']))
    sub_park_name_to_code = dict(zip(original_df['Sub_Science_Park'], cleaned_df['Sub_Science_Park']))
    month_name_map = dict(zip(range(1, 13), ["一月", "二月", "三月", "四月", "五月", "六月", "七月", "八月", "九月", "十月", "十一月", "十二月"]))
    
    park_order = ["新竹科學園區", "中部科學園區", "南部科學園區"]
    available_parks = [park for park in park_order if park in original_df['Science_Park'].unique()]
    
    st.sidebar.header("⚙️ 參數設定")
    
    # Store previous model selection to detect changes
    if 'previous_model_selection' not in st.session_state:
        st.session_state.previous_model_selection = st.session_state.model_selection
    
    model_options = list(MODEL_NAME_MAPPING.keys()) # Use Chinese names for UI
    selected_model_name_chinese = st.sidebar.selectbox(
        "🤖 **模型選擇**", 
        model_options, 
        key='model_selection'
        # on_change callback removed to implement manual check
    )

    # Manually check for changes to fix dropdown bug, avoiding on_change
    if selected_model_name_chinese != st.session_state.previous_model_selection:
        st.session_state.previous_model_selection = selected_model_name_chinese
        clear_results_and_update_temp()
        st.rerun()

    selected_model_name_english = MODEL_NAME_MAPPING[selected_model_name_chinese]


    # Dynamic Hyperparameter Controls
    model_hyperparams = {}
    if selected_model_name_chinese == "決策樹迴歸 (Decision Tree Regression)":
        st.sidebar.markdown("---")
        st.sidebar.subheader("樹模型參數")
        model_hyperparams['max_depth'] = st.sidebar.slider("最大深度 (max_depth)", 1, 20, st.session_state.dt_max_depth, key='dt_max_depth', on_change=clear_results_and_update_temp)
    elif selected_model_name_chinese == "梯度提升樹迴歸 (HistGradient)":
        st.sidebar.markdown("---")
        st.sidebar.subheader("梯度提升樹模型參數")
        model_hyperparams['learning_rate'] = st.sidebar.number_input("學習率 (learning_rate)", 0.01, 1.0, st.session_state.hgb_learning_rate, step=0.01, format="%.2f", key='hgb_learning_rate', on_change=clear_results_and_update_temp)
        model_hyperparams['max_iter'] = st.sidebar.slider("最大迭代次數 (max_iter)", 50, 500, st.session_state.hgb_max_iter, key='hgb_max_iter', on_change=clear_results_and_update_temp)
    elif selected_model_name_chinese == "支持向量迴歸 (SVR)":
        st.sidebar.markdown("---")
        st.sidebar.subheader("SVR 模型參數")
        model_hyperparams['C'] = st.sidebar.number_input("正規化參數 (C)", 0.1, 1000.0, st.session_state.svr_c, step=0.1, key='svr_c', on_change=clear_results_and_update_temp)
        model_hyperparams['gamma'] = st.sidebar.number_input("核函數係數 (gamma)", 0.001, 1.0, st.session_state.svr_gamma, step=0.001, format="%.3f", key='svr_gamma', on_change=clear_results_and_update_temp)
    elif selected_model_name_chinese == "梯度提升樹迴歸 (LightGBM)":
        st.sidebar.markdown("---")
        st.sidebar.subheader("LightGBM 模型參數")
        model_hyperparams['n_estimators'] = st.sidebar.slider("估計器數量 (n_estimators)", 10, 500, st.session_state.lgbm_n_estimators, key='lgbm_n_estimators', on_change=clear_results_and_update_temp)
        model_hyperparams['learning_rate'] = st.sidebar.number_input("學習率 (learning_rate)", 0.01, 0.5, st.session_state.lgbm_learning_rate, step=0.01, format="%.2f", key='lgbm_learning_rate', on_change=clear_results_and_update_temp)


    # Science Park Selection
    st.sidebar.markdown("---")
    st.sidebar.selectbox(
        "🌳 **科學園區**", 
        available_parks, 
        key='selected_park', 
        on_change=on_park_change
    )

    # Sub-Science Park and other dependent widgets
    sub_park_names = get_sub_park_options(st.session_state.get('selected_park'))
    
    if sub_park_names:
        st.sidebar.selectbox(
            "🌲 **子園區**", 
            sub_park_names, 
            key='selected_sub_park', 
            on_change=clear_results_and_update_temp
        )
        
        # Display location info
        if st.session_state.get('selected_sub_park'):
            location_info_row = original_df[original_df['Sub_Science_Park'] == st.session_state.selected_sub_park].iloc[0]
            st.sidebar.markdown(f"""
            - **縣市:** {location_info_row['County']}
            - **鄉鎮市:** {location_info_row['Town']}
            - **氣溫測站:** {location_info_row['Temp_Station_Name']}
            """)
        
        # Year and Month Selection
        st.sidebar.text_input(
            "📅 **西元年**",
            key='selected_year',
            max_chars=4,
            on_change=clear_results_and_update_temp
        )
        st.sidebar.selectbox(
            "🗓️ **月份**", 
            options=list(month_name_map.keys()), 
            format_func=lambda x: month_name_map[x], 
            key='selected_month', 
            on_change=clear_results_and_update_temp,
        )

        # Avg_Temperature Input
        st.sidebar.number_input("🌡️ **月均溫(°C)**", format="%.2f", key='avg_temp', on_change=lambda: st.session_state.update(prediction_result=None, model_metrics=None))
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            predict_button = st.button("💡 **預估**")
        with col2:
            st.button("🧹 **清除**", on_click=lambda: reset_prediction_form(original_df))

        # Prediction Logic
        if predict_button and st.session_state.get('selected_sub_park'):
            try:
                selected_year_int = int(st.session_state.selected_year)
                is_valid_year = True
            except (ValueError, TypeError):
                is_valid_year = False

            if not is_valid_year:
                st.error("請輸入有效的 4 位數西元年份。")
            else:
                st.session_state.prediction_result = None
                st.session_state.model_metrics = None
                
                # Use the preprocessor's new method to train and evaluate
                with st.spinner(f"正在訓練 {selected_model_name_chinese} 模型並預測..."):
                    try:
                        trained_model, metrics, _, _, _ = preprocessor.train_predict_evaluate_model(
                            selected_model_name_english, # Pass English model name to backend
                            X_full, # Pass full X for training/splitting
                            y_full, # Pass full y for training/splitting
                            test_size=0.2,
                            random_state=42,
                            **model_hyperparams # Pass dynamic hyperparameters
                        )
                        st.session_state.trained_model = trained_model
                        st.session_state.model_metrics = metrics

                        # Prepare prediction input for the trained model
                        selected_park_code = park_name_to_code[st.session_state.selected_park]
                        selected_sub_park_code = sub_park_name_to_code[st.session_state.selected_sub_park]
                        
                        original_county_name = original_df[original_df['Sub_Science_Park'] == st.session_state.selected_sub_park]['County'].iloc[0]
                        original_town_name = original_df[original_df['Sub_Science_Park'] == st.session_state.selected_sub_park]['Town'].iloc[0]
                        encoded_county = preprocessor.encoders['County'].index(original_county_name) if original_county_name in preprocessor.encoders['County'] else -1 # Fallback
                        encoded_town = preprocessor.encoders['Town'].index(original_town_name) if original_town_name in preprocessor.encoders['Town'] else -1 # Fallback

                        # Construct single row for prediction input (must match training features order)
                        prediction_input_df = pd.DataFrame([[
                            selected_year_int, st.session_state.selected_month, st.session_state.avg_temp,
                            selected_park_code, selected_sub_park_code,
                            encoded_county, encoded_town
                        ]], columns=features)
                        
                        # Scale the prediction input using the *fitted* scaler from preprocessor
                        # preprocessor.scaler is fitted inside train_predict_evaluate_model
                        prediction_input_scaled = preprocessor.scaler.transform(prediction_input_df)

                        predicted_usage = trained_model.predict(prediction_input_scaled)[0]
                        
                        # Get actual usage if available
                        historical_data = cleaned_df[
                            (cleaned_df['Sub_Science_Park'] == selected_sub_park_code) &
                            (cleaned_df['Year_EN'] == selected_year_int) &
                            (cleaned_df['Month_NUM'] == st.session_state.selected_month)
                        ]
                        actual_usage = historical_data['Electricity_Usage'].iloc[0] if not historical_data.empty else None
                        
                        st.session_state.prediction_result = {"predicted": predicted_usage, "actual": actual_usage}

                    except Exception as e:
                        st.error(f"模型訓練或預測時發生錯誤: {e}")
    else:
        st.sidebar.warning("此科學園區下沒有可用的子園區。")


    # --- Results Display Area ---
    if st.session_state.get('prediction_result'):
        result = st.session_state.prediction_result
        predicted_usage = result['predicted']
        actual_usage = result.get('actual') # From historical data or previous input

        st.subheader("🔮 預測結果")
        
        # Create columns for prediction info and gauge
        col_info, col_gauge = st.columns([1, 1], gap="large")

        with col_info:
            st.markdown(f"<h3>📈 用電量預估值 (萬KW)：<strong style='color: #FF8C00;'>{predicted_usage:,.2f}</strong></h3>", unsafe_allow_html=True)
            
            # Display historical actual usage if available, otherwise ask for manual input
            if result.get('actual') is not None: # Use result.get('actual') from history
                st.markdown(f"<h3>⚡ 實際用電量 (萬KW)：<strong>{result['actual']:,.2f}</strong></h3>", unsafe_allow_html=True)
            else:
                st.markdown("<h4>⚡ 請輸入實際用電量 (萬KW，選填)</h4>", unsafe_allow_html=True)
                # Initialize actual_usage_input in session_state if it doesn't exist.
                if 'actual_usage_input' not in st.session_state:
                    st.session_state.actual_usage_input = 0.0 # Set initial value
                
                # Display the widget. Its value will be automatically updated in st.session_state.actual_usage_input
                st.number_input(" ", min_value=0.0, format="%.2f", key="actual_usage_input", label_visibility="collapsed")
                
                # Access the value directly from session state after the widget has been rendered
                if st.session_state.actual_usage_input > 0:
                    actual_usage = st.session_state.actual_usage_input # Use user input as actual_usage
        
        # --- Gauge and Textual Feedback ---
        # Only show gauge and feedback if we have an actual_usage (either historical or user-provided)
        if actual_usage is not None and actual_usage > 0:
            diff_percent = abs(predicted_usage - actual_usage) / actual_usage * 100
            
            with col_gauge:
                fig = create_gauge_chart(diff_percent)
                st.plotly_chart(fig, use_container_width=True)

            # Re-introduce rounding for robust zero-check to fix balloons
            predicted_rounded = round(predicted_usage, 2)
            actual_rounded = round(actual_usage, 2)
            if predicted_rounded == actual_rounded:
                diff_percent = 0.0

            # New user-requested feedback text
            if diff_percent == 0.0:
                st.success("That's awesome! 預測完全準確！")
                st.balloons()
            elif diff_percent <= 3:
                st.success("😊 誤差率低於 3%，太棒了！感謝您的正面回饋")
            elif diff_percent <= 5: # This now correctly means >3% and <=5%
                st.info("😐 誤差率介於 3% ~ 5%，感謝您的回饋，我們會繼續努力👍")
            else: # This covers > 5%
                st.warning("😥 Oh no！誤差率高於 5%，感謝您的回饋，我們會參考這項資訊來改進模型")
            
            st.markdown("---") 
    
    # --- Model Performance Evaluation Display ---
    if st.session_state.get('model_metrics'):
        st.subheader("🎯 模型效能評估 (Model Performance Evaluation)")
        metrics = st.session_state.model_metrics
        
        col_r2, col_rmse, col_mae = st.columns(3)
        with col_r2:
            st.metric(
                label="R-squared (R²)", 
                value=f"{metrics['R2']:.4f}",
                help="R-squared 值愈接近 1 表示模型解釋目標變異的能力愈強，擬合效果愈好。"
            )
        with col_rmse:
            st.metric(
                label="均方根誤差 (RMSE)", 
                value=f"{metrics['RMSE']:.2f}",
                help="RMSE 衡量預測值與實際值之間的平均偏差，單位與目標變數相同，值愈小表示模型預測愈準確。"
            )
        with col_mae:
            st.metric(
                label="平均絕對誤差 (MAE)", 
                value=f"{metrics['MAE']:.2f}",
                help="MAE 衡量預測值與實際值之間絕對誤差的平均值，單位與目標變數相同，值愈小表示模型預測愈準確。"
            )
        
        st.markdown(f"""
        <div style="font-size: 0.9em; color: gray;">
        * 模型 '{selected_model_name_chinese}' 在測試集上的效能。
        * 測試集大小佔總資料集的 20%。
        </div>
        """, unsafe_allow_html=True)