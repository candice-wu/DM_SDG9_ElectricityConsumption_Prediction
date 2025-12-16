import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import altair as alt
from src.data_preprocessing import DataPreprocessor
from src.ui_components import render_app_info, render_data_status
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="模型比較分析", page_icon="🎭", layout="wide")

# --- Constants & Helper Functions ---
MODEL_NAME_MAPPING = {
    "線性迴歸 (Linear Regression)": "LinearRegression",
    "決策樹迴歸 (Decision Tree Regression)": "DecisionTreeRegressor",
    "梯度提升樹迴歸 (HistGradient)": "HistGradientBoostingRegressor",
    "支持向量迴歸 (SVR)": "SVR",
    "梯度提升樹迴歸 (LightGBM)": "LGBMRegressor"
}

# --- Colors ---
HIGHLIGHT_COLOR = "#4481D7"
MULTI_MODEL_COLORS = ['#3ABBDE', '#DD5C6A', '#F5C65D', '#D96015', '#9FCE63']
SINGLE_MODEL_COLOR = '#BC72A7'
FEATURE_IMPORTANCE_COLOR = '#F5C65D'


def get_styled_text(text):
    return f'<span style="color:{HIGHLIGHT_COLOR}; font-weight:bold;">{text}</span>'


# --- Main App ---
st.title("🎭 模型比較與進階分析")

# --- Data Loading and Caching ---
def load_data():
    if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
        return None, None, None
    cleaned_df = st.session_state['cleaned_df']
    preprocessor = st.session_state['preprocessor']
    return cleaned_df, preprocessor, st.session_state.get('original_df')


@st.cache_resource(show_spinner="正在為所有模型訓練並快取結果...")
def get_all_models_data(_preprocessor, _X_full, _y_full, models_dict):
    all_models_data = {}
    for model_name_chinese, model_name_english in models_dict.items():
        try:
            # Correctly unpack the 5 values returned from the refactored function
            model, metrics, X_test_scaled, y_test, y_pred = _preprocessor.train_predict_evaluate_model(
                model_name_english, _X_full, _y_full, test_size=0.2, random_state=42
            )
            # Store all the returned data, not just model and predictions
            all_models_data[model_name_chinese] = {
                "model": model,
                "metrics": metrics,
                "predictions": y_pred,
                "X_test_scaled": X_test_scaled,
                "y_test": y_test
            }
        except Exception as e:
            st.warning(f"無法為模型 '{model_name_chinese}' 產生結果：{e}")
            all_models_data[model_name_chinese] = None # Mark failed model
    return all_models_data


cleaned_df, preprocessor, original_df = load_data()

if cleaned_df is None or preprocessor is None:
    st.warning("⬅️ 請先至「📄 資料探索與清理」頁面上傳並清理資料")
    st.stop()

# Render sidebar elements
render_app_info()
render_data_status(cleaned_df)

st.info("""
此頁面提供一個互動式儀表板，用於深入比較和診斷在「用電量預測」頁面上訓練的各個迴歸模型之效能與行為
- **預測值 vs. 實際值圖**：直觀地評估模型的整體準確性和潛在偏差
- **殘差圖**：用於診斷模型的系統性錯誤，理想的殘差應隨機分佈
- **特徵重要性圖**：揭示模型在進行預測時最依賴哪些特徵
- **混淆矩陣分析**：將連續預測值轉換為級距，評估模型在各級距上的分類準確度
""")

X_full, y_full, feature_names, _ = preprocessor.get_prediction_data(cleaned_df)
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42
)
all_models_data = get_all_models_data(preprocessor, X_full, y_full, MODEL_NAME_MAPPING)


# --- Plotting Functions ---
def plot_prediction_vs_actual(models_data, y_true_series, selected_models, colors):
    fig = go.Figure()
    for i, model_name in enumerate(selected_models):
        if model_name in models_data and 'predictions' in models_data[model_name]:
            y_pred = models_data[model_name]['predictions']
            fig.add_trace(go.Scatter(
                x=y_true_series,
                y=y_pred,
                mode='markers',
                name=model_name,
                marker=dict(opacity=0.7, color=colors[i % len(colors)])
            ))

    if models_data:
        preds = [
            md['predictions']
            for md in models_data.values()
            if md.get('predictions') is not None and md['predictions'].size > 0
        ]
        if preds:
            min_val = min(y_true_series.min(), *(p.min() for p in preds))
            max_val = max(y_true_series.max(), *(p.max() for p in preds))
            fig.add_shape(
                type='line',
                x0=min_val, y0=min_val,
                x1=max_val, y1=max_val,
                line=dict(color='Gray', width=3, dash='dash')
            )

    fig.update_layout(
        title_text="預測值 vs. 實際值 (Prediction vs. Actual)",
        xaxis_title="實際用電量 (Actual Usage)",
        yaxis_title="預測用電量 (Predicted Usage)",
        legend_title="模型",
        height=600
    )
    return fig


def plot_residuals(models_data, y_true_series, selected_models, colors):
    fig = go.Figure()
    for i, model_name in enumerate(selected_models):
        if model_name in models_data and 'predictions' in models_data[model_name]:
            y_pred = models_data[model_name]['predictions']
            residuals = y_true_series - y_pred
            fig.add_trace(go.Scatter(
                x=y_pred,
                y=residuals,
                mode='markers',
                name=model_name,
                marker=dict(opacity=0.7, color=colors[i % len(colors)])
            ))

    fig.add_hline(y=0, line_width=3, line_dash="dash", line_color="Gray")
    fig.update_layout(
        title_text="殘差圖 (Residuals Plot)",
        xaxis_title="預測用電量 (Predicted Usage)",
        yaxis_title="殘差 (Actual - Predicted)",
        legend_title="模型",
        height=500
    )
    return fig


def plot_feature_importance(model_obj, features, color, X_test_scaled=None, y_test=None):
    importance = None
    if isinstance(model_obj, HistGradientBoostingRegressor):
        if X_test_scaled is not None and y_test is not None:
            with st.spinner("正在計算排列重要性..."):
                result = permutation_importance(model_obj, X_test_scaled, y_test, n_repeats=10, random_state=42)
                importance = result.importances_mean
        else:
            return None, None # Cannot calculate without test data
    elif hasattr(model_obj, 'feature_importances_'):
        importance = model_obj.feature_importances_
    elif hasattr(model_obj, 'coef_'):
        importance = np.abs(model_obj.coef_.flatten())

    if importance is not None and len(importance) == len(features):
        df = pd.DataFrame({
            'Feature': features,
            'Importance': importance
        }).sort_values('Importance', ascending=False)

        chart = alt.Chart(df).mark_bar(color=color).encode(
            x=alt.X('Importance', title='重要性分數'),
            y=alt.Y('Feature', sort='-x', title='特徵')
        ).properties(title='特徵重要性')

        return chart, df.head(3)

    return None, None


def discretize_data(y_true, num_bins):
    try:
        _, bin_edges = pd.qcut(y_true, q=num_bins, retbins=True, duplicates='drop')

        if num_bins == 2:
            prefixes = ['低', '高']
        elif num_bins == 3:
            prefixes = ['低', '中', '高']
        elif num_bins == 4:
            prefixes = ['低', '中低', '中高', '高']
        else:
            prefixes = ['極低', '低', '中', '高', '極高']

        bin_labels = [
            f"{prefixes[i]}用量 ({bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}]"
            for i in range(len(bin_edges) - 1)
        ]

        y_true_discrete = pd.cut(
            y_true,
            bins=bin_edges,
            labels=bin_labels,
            include_lowest=True,
            right=True
        )
        return y_true_discrete, bin_labels, bin_edges
    except Exception:
        return None, [], []


def plot_confusion_matrix(cm, labels):
    fig = px.imshow(
        cm,
        labels=dict(x="預測級距", y="實際級距", color="次數"),
        x=labels, y=labels,
        text_auto=True,
        color_continuous_scale=px.colors.diverging.Spectral_r
    )
    fig.update_layout(
        title_text='混淆矩陣',
        height=500,
        xaxis={'side': 'top', 'tickangle': -45}
    )
    return fig


# --- UI Rendering ---
st.header("🎨 多模型效能視覺化比較")
model_options = list(all_models_data.keys())

if not model_options:
    st.error("無任何模型成功載入，請檢查 `get_all_models_data` 函式。")
    st.stop()

selected_models = st.multiselect("選擇要比較的模型：", options=model_options, default=model_options)

if selected_models:
    colors = [SINGLE_MODEL_COLOR] if len(selected_models) == 1 else MULTI_MODEL_COLORS

    st.subheader("🎏 預測值 vs. 實際值分析")
    fig1 = plot_prediction_vs_actual(all_models_data, y_test, selected_models, colors)
    st.plotly_chart(fig1, use_container_width=True)

    with st.expander("📊 結論：預測值 vs. 實際值分析", expanded=True):
        metrics = {
            name: {
                "R²": r2_score(y_test, all_models_data[name]['predictions']),
                "RMSE": np.sqrt(mean_squared_error(y_test, all_models_data[name]['predictions']))
            }
            for name in selected_models
            if name in all_models_data and all_models_data[name]
        }

        if metrics:
            metrics_df = pd.DataFrame(metrics).T.sort_values("R²", ascending=False)
            st.dataframe(
                metrics_df.style
                .format("{:.4f}")
                .highlight_max(axis=0, subset="R²", color="#9ACD32")
                .highlight_min(axis=0, subset="RMSE", color="#F08080")
            )

            best_r2_model = metrics_df["R²"].idxmax()
            best_rmse_model = metrics_df["RMSE"].idxmin()

            st.markdown("---")
            st.markdown(f"""
            **結論分析：**<br>
            - 完美的模型，其數據點會落在 {get_styled_text("45 度的對角線")} 上
            - {get_styled_text('R²')} 愈接近 {get_styled_text('1')} 表示模型解釋變異能力愈{get_styled_text("強")}，{get_styled_text('RMSE')} 愈{get_styled_text('小')}表示預測誤差愈{get_styled_text("低")}
            - R² 表現最好的模型為 **{get_styled_text(best_r2_model)}** (R² = {metrics_df.loc[best_r2_model, 'R²']:.4f})
            - RMSE 表現最好的模型為 **{get_styled_text(best_rmse_model)}** (RMSE = {metrics_df.loc[best_rmse_model, 'RMSE']:.4f})
            - R² 和 RMSE 綜合表現最佳的模型為 **{get_styled_text(best_r2_model)}**
            """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("🧩 殘差分析")
    fig2 = plot_residuals(all_models_data, y_test, selected_models, colors)
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("📊 結論：殘差分析", expanded=True):
        res_stats = {
            name: {
                "平均值": (y_test - all_models_data[name]['predictions']).mean(),
                "標準差": (y_test - all_models_data[name]['predictions']).std()
            }
            for name in selected_models
            if name in all_models_data and all_models_data[name]
        }

        if res_stats:
            res_stats_df = pd.DataFrame(res_stats).T.sort_values("標準差")

            st.dataframe(
                res_stats_df.style
                .format("{:.4f}")
                .highlight_min(axis=0, subset="標準差", color="#F08080")
                .apply(lambda x: ['background-color: #9ACD32' if abs(v) < 1e-3 else '' for v in x],
                       subset=['平均值'])
            )

            best_std_model = res_stats_df["標準差"].idxmin()

            st.markdown("---")
            st.markdown(f"""
            **結論分析：**<br>
            - 理想的殘差圖中，資料點應隨機分佈在{get_styled_text('零點水平線')}周圍，沒有明顯的模式或趨勢，有助於減少系統性偏差
            - {get_styled_text('殘差平均值')}應接近 {get_styled_text('零')}，{get_styled_text('標準差')} 愈 {get_styled_text('小')}，表示預測更 {get_styled_text('穩定')}  
            - 目前殘差標準差最小的模型為 {get_styled_text(best_std_model)} 標準差 = {res_stats_df.loc[best_std_model, '標準差']:.4f}
            """, unsafe_allow_html=True)

st.markdown("---")

# --- Single Model Analysis ---
st.header("🪐 單一模型深度分析")
col1, col2 = st.columns(2)

with col1:
    st.subheader("⭐ 特徵重要性分析")
    fi_options = [
        name for name, data in all_models_data.items()
        if data and (
            hasattr(data.get('model'), 'feature_importances_') or 
            hasattr(data.get('model'), 'coef_') or
            isinstance(data.get('model'), HistGradientBoostingRegressor)
        )
    ]

    if not fi_options:
        st.warning("⚠️ 目前載入的模型均不支援直接的特徵重要性分析。")
    else:
        model_fi = st.selectbox("選擇模型以分析特徵重要性：", options=fi_options, key="fi_model")
        if model_fi and model_fi in all_models_data:
            model_info = all_models_data[model_fi]
            # We need the specific test set used for this model, which is now stored
            X_test_scaled_model = model_info.get("X_test_scaled")
            y_test_model = model_info.get("y_test")

            if X_test_scaled_model is not None and y_test_model is not None:
                chart, top_feats = plot_feature_importance(
                    model_info["model"],
                    feature_names,
                    FEATURE_IMPORTANCE_COLOR,
                    X_test_scaled_model,
                    y_test_model
                )
                if chart is not None:
                    st.altair_chart(chart, use_container_width=True)
                    with st.expander("📊 結論：特徵重要性分析", expanded=True):
                        st.markdown("##### 方法定義")
                        st.markdown(f"""
                            - 此分析有助於理解模型的決策過程，並可用於特徵選擇與工程
                                - 各特徵的相對重要性會影響模型進行預測的程度
                                - 重要性分數愈 {get_styled_text('高')} 則影響愈 {get_styled_text('大')}，有助於理解預測邏輯與關鍵因子
                            - 對於 {get_styled_text('線性迴歸 (Linear Regression)')} 模型，使用 {get_styled_text('係數絕對值')} 來評估特徵影響力
                            - 以下三大模型，使用 {get_styled_text('內建特徵重要性 (Feature Importances)')} 屬性來評估特徵影響力
                                - {get_styled_text('決策樹迴歸 (Decision Tree Regression)')}
                                - {get_styled_text('梯度提升樹迴歸 (LightGBM)')}
                                - {get_styled_text('梯度提升樹迴歸 (HistGradient)')}
                                - 內建特徵重要性基於特徵在樹結構中分裂節點的貢獻度計算
                            - {get_styled_text('梯度提升樹迴歸 (HistGradient)')} 使用 {get_styled_text('排列重要性 (Permutation Importance)')} 方法評估特徵影響力
                                - 排列重要性透過隨機打亂每個特徵的值，觀察模型預測性能的變化來評估該特徵的重要性
                            - 對於 {get_styled_text('支持向量迴歸 (SVR)')} 模型，因為使用{get_styled_text('非線性核心')}，無法直接評估特徵重要性
                        """, unsafe_allow_html=True)

                        st.markdown("---")
                        st.markdown("##### 數據特性")
                        st.markdown(f"此模型 **{get_styled_text(model_fi)}** 最重視的前三大特徵為：", unsafe_allow_html=True)

                        for _, feat in top_feats.iterrows():
                            st.markdown(
                                f"- **{get_styled_text(feat['Feature'])}** (重要性分數 = {feat['Importance']:.4f})",
                                unsafe_allow_html=True
                            )
                else:
                    st.info(f"模型「{model_fi}」不提供直接的特徵重要性屬性（例如：SVR 非線性核心）。")
            else:
                st.error("模型資料不完整，缺少進行特徵重要性分析所需的測試集。")

with col2:
    st.subheader("🧮 級距預測準確度評估 (混淆矩陣)")
    model_cm = st.selectbox("選擇模型以產生混淆矩陣：", options=model_options, key="cm_model")
    num_bins = st.slider("選擇用電量級距數量 (分位數)：", 2, 5, 3, key="cm_bins")

    if model_cm and model_cm in all_models_data:
        y_true_discrete, bin_labels, bin_edges = discretize_data(y_test, num_bins)
        if y_true_discrete is not None and not y_true_discrete.empty:
            y_pred = all_models_data[model_cm]['predictions']

            # same discretization rule for prediction
            y_pred_discrete = pd.cut(
                y_pred,
                bins=bin_edges,
                labels=bin_labels,
                include_lowest=True,
                right=True
            )

            # Handle predictions out of range
            out_of_range_label = "預測超出範圍"
            has_out_of_range = y_pred_discrete.isnull().any()

            final_labels = bin_labels.copy()
            if has_out_of_range:
                y_pred_discrete = y_pred_discrete.add_categories(out_of_range_label).fillna(out_of_range_label)
                if out_of_range_label not in final_labels:
                    final_labels.append(out_of_range_label)
            
            # Ensure y_pred_discrete does not contain categories not in y_true_discrete unless it's the out_of_range_label
            y_pred_discrete = y_pred_discrete.astype(pd.CategoricalDtype(categories=final_labels))


            cm = confusion_matrix(y_true_discrete, y_pred_discrete, labels=final_labels)
            fig3 = plot_confusion_matrix(cm, final_labels)
            st.plotly_chart(fig3, use_container_width=True)

            with st.expander("📊 結論：混淆矩陣分析", expanded=True):
                # Calculate accuracy excluding out-of-range predictions if they exist
                valid_indices = y_pred_discrete != out_of_range_label
                accuracy = accuracy_score(y_true_discrete[valid_indices], y_pred_discrete[valid_indices]) if valid_indices.any() else 0

                st.markdown("##### 方法定義")
                st.markdown(f"""
                            - 混淆矩陣展示模型在不同「用電量級距」的分類表現
                            - 每個格子中的數值代表模型預測落在該級距的次數
                                - 對角線上的數值代表正確預測
                                - 非對角線則為錯誤預測
                            - 準確度計算方式為：正確預測次數 / 總預測次數
                            """, unsafe_allow_html=True
                            )                
                if has_out_of_range:
                    st.markdown(f"""
                                - 關於「{get_styled_text(out_of_range_label)}」：
                                    - 線性迴歸、梯度提升等模型可能會外插 (extrapolation)，導致預測值超出訓練資料的範圍
                                    - 此模型 {get_styled_text(model_cm)} 代表預測值超出測試資料的分位數範圍
                                    """, unsafe_allow_html=True)


                st.markdown("---")
                st.markdown("##### 數據特性")
                st.markdown(f"""
                            此模型 {get_styled_text(model_cm)} 整體準確度為 **{get_styled_text(f"{accuracy:.4%}")}**
                            """, unsafe_allow_html=True
                            )

        else:
            st.error(f"無法將資料離散化為 {num_bins} 個級距，請嘗試不同的級距數量")

st.markdown("---")
st.success("所有模型分析功能已建構完成！")