import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import colorsys
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.tree import DecisionTreeClassifier
from src.ui_components import render_app_info, render_data_status

st.set_page_config(page_title="資料轉換", page_icon="♻️", layout="wide")

st.title("♻️ 資料轉換 (Data Transformation)")

# Inject custom CSS for info box
st.markdown("""
    <style>
    .info-box {
        background-color: #e7f3ff;
        border-left: 5px solid #4481d7;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        color: #0c2b51;
    }
    .info-box ul {
        margin-bottom: 0;
    }
    </style>
    """, unsafe_allow_html=True)


# --- Helper Functions ---

@st.cache_data
def get_transformed_data(_preprocessor):
    return _preprocessor.transform_data()

def get_df_statistics(df, column):
    """Calculates a comprehensive set of statistics for a given df and column."""
    stats = {}
    data = df[column].dropna()
    
    stats['Mean'] = data.mean()
    stats['Median'] = data.median()
    stats['Mode'] = data.mode().iloc[0] if not data.mode().empty else 'N/A'
    
    five_num = data.describe(percentiles=[.25, .5, .75])
    stats['Min'] = five_num['min']
    stats['Q1'] = five_num['25%']
    stats['Q2 (Median)'] = five_num['50%']
    stats['Q3'] = five_num['75%']
    stats['Max'] = five_num['max']
    
    stats['Variance'] = data.var()
    stats['Standard Deviation'] = data.std()
    
    Q1 = stats['Q1']
    Q3 = stats['Q3']
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    stats['Outliers'] = len(outliers)
    
    return stats

def plot_distribution(df, column, color, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    r, g, b = plt.cm.colors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    darker_color = colorsys.hls_to_rgb(h, l * 0.5, s)
    sns.histplot(df[column], kde=False, ax=ax, color=color, stat='density')
    sns.kdeplot(df[column], ax=ax, color=darker_color, linewidth=2.5)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    return fig

def plot_boxplot(df, column, color, title):
    fig, ax = plt.subplots(figsize=(6, 2))
    sns.boxplot(x=df[column], ax=ax, color=color)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Value")
    return fig

def plot_discretized_data(data_series, original_column_name, color, title, order=None):
    fig, ax = plt.subplots(figsize=(8, 5)) # Increased figure size for better label visibility
    sns.countplot(x=data_series, ax=ax, palette=[color], order=order)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(original_column_name + ' (離散化區間)', fontsize=12)
    ax.set_ylabel("計數", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plot_clustering_validation(df, cluster_col, target_col, color, order=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x=cluster_col, y=target_col, data=df, ax=ax, color=color, order=order)
    ax.set_title(f'各聚類區間的用電量分佈', fontsize=16)
    ax.set_xlabel('聚類分析產生的區間', fontsize=12)
    ax.set_ylabel('用電量 (萬KW)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plot_smoothing_results(original_series, smoothed_series, title, original_color, smoothed_color):
    """繪製原始資料與平滑後資料的對比圖"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(original_series.index, original_series, color=original_color, label='原始資料', alpha=0.6)
    ax.plot(smoothed_series.index, smoothed_series, color=smoothed_color, label='平滑後資料', linewidth=2)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("資料點索引", fontsize=12)
    ax.set_ylabel("數值", fontsize=12)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    return fig
    
def display_statistics_table(stats_dict):
    with st.expander("🔎 查看所有統計指標"):
        stats_df = pd.DataFrame(list(stats_dict.items()), columns=['指標 (Measure)', '數值 (Value)'])
        stats_df['數值 (Value)'] = stats_df['數值 (Value)'].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
        st.table(stats_df)

def generate_conclusion(stats):
    mean = stats['Mean']
    median = stats['Median']
    std_dev = stats['Standard Deviation']
    outliers = stats['Outliers']
    
    blue = "#4481D7"

    if abs(mean - median) / (mean or 1) < 0.1:
        skewness = f"平均值 (<span style='color:{blue}'>**{mean:.4f}**</span>) 與中位數 (<span style='color:{blue}'>**{median:.4f}**</span>) 非常接近，表示 **分佈相對對稱**"
    elif mean > median:
        skewness = f"平均值 (<span style='color:{blue}'>**{mean:.4f}**</span>) 大於中位數 (<span style='color:{blue}'>**{median:.4f}**</span>)，表示資料呈現 **右偏分佈** (有少數極大值)"
    else:
        skewness = f"平均值 (<span style='color:{blue}'>**{mean:.4f}**</span>) 小於中位數 (<span style='color:{blue}'>**{median:.4f}**</span>)，表示資料呈現 **左偏分佈** (有少數極小值)"

    dispersion = f"標準差為 <span style='color:{blue}'>**{std_dev:.4f}**</span>，這代表資料點圍繞平均值的離散程度"

    if outliers > 0:
        outlier_text = f"偵測到 <span style='color:{blue}'>**{outliers}**</span> 個離群值，這些顯著高於或低於中心的數值可能會對某些分析模型產生影響"
    else:
        outlier_text = "未偵測到明顯的離群值"

    conclusion = f"""
    - **集中趨勢**：{skewness}
    - **離散程度**：{dispersion}
    - **離群值**：{outlier_text}
    """
    return conclusion

def generate_discretization_conclusion(discretized_series, method_name, original_col_name):
    blue = "#4481D7"
    num_bins = len(discretized_series.cat.categories)
    bin_counts = discretized_series.value_counts().sort_index()
    
    conclusion = f"""
- **裝箱方法**：使用 <span style='color:{blue}'>****{method_name}****</span> 方法將原始資料 <span style='color:{blue}'>****{original_col_name}****</span> 離散化為 <span style='color:{blue}'>**{num_bins}**</span> 個區間
- **分佈概覽**：
"""
    for bin_label, count in bin_counts.items():
        conclusion += f"\n    - 區間 <span style='color:{blue}'>**{bin_label}**</span> 包含 <span style='color:{blue}'>**{count}**</span> 個資料點"
    
    return conclusion

def generate_clustering_conclusion(df, cluster_col, target_col):
    blue = "#4481D7"
    
    # Ensure the groupby operation respects the categorical order for sorting
    if pd.api.types.is_categorical_dtype(df[cluster_col]):
        # The categories are already sorted from the preprocessor
        median_values = df.groupby(cluster_col, observed=True)[target_col].median()
    else:
        # Fallback for non-categorical, though the preprocessor should handle this
        median_values = df.groupby(cluster_col)[target_col].median().sort_index()

    blue = "#4481D7"
    
    # 1. Generate the dynamic insight
    insight_text = ""
    # Check for monotonicity
    if median_values.is_monotonic_increasing:
        insight_text = f"<b>洞見</b>：聚類分析發現一個<span style='color:{blue}'><b>類線性關係</b></span>，目標變數（用電量）隨著特徵區間的增加而穩定上升"
    elif median_values.is_monotonic_decreasing:
        insight_text = f"<b>洞見</b>：聚類分析發現一個<span style='color:{blue}'><b>類線性關係</b></span>，目標變數（用電量）隨著特徵區間的增加而穩定下降"
    else:
        insight_text = f"<b>洞見</b>：聚類分析揭示一個有趣的<span style='color:{blue}'><b>非線性關係</b></span>，用電量並非簡單地隨特徵遞增或遞減，而是在特定區間出現高峰或低谷"

    # 2. Basic statistics (min/max)
    if len(median_values) > 1:
        min_cluster = median_values.idxmin()
        max_cluster = median_values.idxmax()
        stats_text = f"""
- **用電量最低區間**：<span style='color:{blue}'>**{min_cluster}**</span>，中位數用電量為 <span style='color:{blue}'>**{median_values.min():.4f}**</span> 萬KW<br>
- **用電量最高區間**：<span style='color:{blue}'>**{max_cluster}**</span>，中位數用電量為 <span style='color:{blue}'>**{median_values.max():.4f}**</span> 萬KW"""
    else:
        stats_text = "需要多於一個區間來進行比較"
        
    conclusion = f"""
{insight_text}
<br><br>
- **關聯性分析**：盒鬚圖展示了每個由特徵分群產生的區間內，目標變數（用電量）的分佈情況
{stats_text}
"""
    return conclusion

def generate_chi2_conclusion(chi2, p, dof):
    blue = "#4481D7"
    
    conclusion = f"""
- **卡方統計值 (χ²)**：<span style='color:{blue}'>**{chi2:.4f}**</span>
- **p-value**：<span style='color:{blue}'>**{p:.4f}**</span>
- **自由度 (dof)**：<span style='color:{blue}'>**{dof}**</span>
<br>
"""
    if p < 0.05:
        conclusion += f"**結論**：由於 p-value (<span style='color:{blue}'>**{p:.4f}**</span>) **小於** 顯著性水準 0.05，故<span style='color:{blue}'>**拒絕虛無假設**</span>，這兩個離散化後的變數之間存在**顯著的統計關聯性**"
    else:
        conclusion += f"**結論**：由於 p-value (<span style='color:{blue}'>**{p:.4f}**</span>) **大於** 顯著性水準 0.05，故<span style='color:{blue}'>**無法拒絕虛無假設**</span>，這兩個離散化後的變數之間**沒有足夠的證據顯示存在統計關聯性**"
    return conclusion

def generate_smoothing_conclusion(method_name, column_name, bins):
    """生成平滑化方法的動態結論"""
    blue = "#4481D7"
    if "平均值" in method_name:
        method_desc = "將每個分箱中的所有資料點替換為該分箱的**平均值**"
    elif "中位數" in method_name:
        method_desc = "將每個分箱中的所有資料點替換為該分箱的**中位數**，這對離群值的影響較不敏感"
    else: # 邊界
        method_desc = "將每個分箱中的資料點替換為距離其最近的分箱**邊界值**"

    conclusion = f"""
- **平滑化方法**：<span style='color:{blue}'>**{method_name}**</span>
- **目標特徵**：對 <span style='color:{blue}'>**{column_name}**</span> 欄位進行處理
- **方法定義**：此方法首先將資料分成 <span style='color:{blue}'>**{bins}**</span> 個等寬的區間（分箱），然後{method_desc}
- **效果**：從上方的圖表可以看出，這種方法有助於**減少資料中的雜訊**和**突發性波動**，讓底層的趨勢更加明顯
"""
    return conclusion

# --- Main App Logic ---

if 'cleaned_df' not in st.session_state or 'preprocessor' not in st.session_state:
    st.warning("⬅️ 請先至「📄 資料探索與清理」頁面上傳並清理資料" )
    st.stop()

render_app_info()
cleaned_df = st.session_state['cleaned_df']
render_data_status(cleaned_df)

# --- Raw Data Overview Section ---
st.header("🏀 原始資料概覽 (Raw Data Overview)")
st.info("此區塊呈現原始資料（清理後）的敘述性統計與分佈，可以在資料轉換 (平滑資料作收斂) 前先了解其基本特性", icon="ℹ️")
original_df = st.session_state['df']
preprocessor = st.session_state['preprocessor']

overview_col1, overview_col2 = st.columns(2)

with overview_col1:
    st.subheader("⚡ 用電量 (Electricity Consumption)")
    fig_dist_elec = plot_distribution(original_df, 'Electricity_Usage', '#3CBBDE', '原始用電量分佈')
    st.pyplot(fig_dist_elec)
    fig_box_elec = plot_boxplot(original_df, 'Electricity_Usage', '#3CBBDE', '原始用電量盒鬚圖')
    st.pyplot(fig_box_elec)
    elec_stats = preprocessor.get_raw_data_statistics('Electricity_Usage')
    
    with st.expander("📊 結論：用電量原始資料分析"):
        st.markdown(generate_conclusion(elec_stats), unsafe_allow_html=True)
    display_statistics_table(elec_stats)

with overview_col2:
    st.subheader("🌡️ 月均溫 (Average Temperature/Monthly)")
    fig_dist_temp = plot_distribution(original_df, 'Avg_Temperature', '#935AB3', '原始月均溫分佈')
    st.pyplot(fig_dist_temp)
    fig_box_temp = plot_boxplot(original_df, 'Avg_Temperature', '#935AB3', '原始月均溫盒鬚圖')
    st.pyplot(fig_box_temp)
    temp_stats = preprocessor.get_raw_data_statistics('Avg_Temperature')

    with st.expander("📊 結論：月均溫原始資料分析"):
        st.markdown(generate_conclusion(temp_stats), unsafe_allow_html=True)
    display_statistics_table(temp_stats)

st.divider()

# --- Transformed Data Section ---
transformed_df = get_transformed_data(preprocessor)

# --- Normalization Section ---
st.header("⚾ 正規化 (Normalization)")
st.markdown("""
<div class="info-box">
<ul>
    <li>此頁面靜態展示對數值型特徵（Avg_Temperature, Electricity_Usage）套用三種不同正規化方法後的結果與分佈變化</li>
    <li>正規化是將數值特徵縮放到一個通用範圍的過程，例如 <span style='color:#4481D7'><b>[0, 1]</b></span> 或 <span style='color:#4481D7'><b>[-1, 1]</b></span>，而不會扭曲其值的範圍差異</li>
</ul>
</div>
""", unsafe_allow_html=True)


# --- Electricity_Usage ---
st.subheader("⚡ 用電量 (Electricity Consumption)")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("##### 極值正規化 (Min-Max)")
    fig = plot_distribution(transformed_df, "Electricity_Usage_min_max", "#F5C65D", "Min-Max Normalization")
    st.pyplot(fig)
    with st.expander("📊 結論：極值正規化 (Min-Max)"):
        st.markdown("""
        **方法定義**：將資料線性縮放到 <span style='color:#4481D7'>**[0, 1]**</span> 範圍內，對異常值較為敏感，因為最大值和最小值會影響整體縮放結果
        """, unsafe_allow_html=True)
        st.markdown("--- ")
        st.markdown("##### **數據特性**")
        stats = get_df_statistics(transformed_df, "Electricity_Usage_min_max")
        st.markdown(generate_conclusion(stats), unsafe_allow_html=True)
    with st.expander("🔎 查看數據"):
        st.dataframe(transformed_df[["Electricity_Usage", "Electricity_Usage_min_max"]].style.format('{:.4f}'))

with col2:
    st.markdown("##### Z分配標準化 (Z-score)")
    fig = plot_distribution(transformed_df, "Electricity_Usage_z_score", "#45C8C5", "Z-score Normalization")
    st.pyplot(fig)
    with st.expander("📊 結論：Z分配標準化 (Z-score)"):
        st.markdown("""
        **方法定義**：將資料轉換為平均值為 <span style='color:#4481D7'>**0**</span>、標準差為 <span style='color:#4481D7'>**1**</span> 的分佈，適用於需要比較不同尺度特徵或期望資料呈常態分佈的演算法
        """, unsafe_allow_html=True)
        st.markdown("--- ")
        st.markdown("##### **數據特性**")
        stats = get_df_statistics(transformed_df, "Electricity_Usage_z_score")
        st.markdown(generate_conclusion(stats), unsafe_allow_html=True)
    with st.expander("🔎 查看數據"):
        st.dataframe(transformed_df[["Electricity_Usage", "Electricity_Usage_z_score"]].style.format('{:.4f}'))

with col3:
    st.markdown("##### 十進位正規化 (Decimal Scaling)")
    fig = plot_distribution(transformed_df, "Electricity_Usage_decimal_scaled", "#DD6D6A", "Decimal Scaling Normalization")
    st.pyplot(fig)
    with st.expander("📊 結論：十進位正規化 (Decimal Scaling)"):
        st.markdown("""
        **方法定義**：透過移動小數點來實現資料縮放，使其絕對值小於 <span style='color:#4481D7'><b>1</b></span>，縮放因子取決於資料的最大絕對值，是一種簡單的正規化方法
        """, unsafe_allow_html=True)
        st.markdown("--- ")
        st.markdown("##### **數據特性**")
        stats = get_df_statistics(transformed_df, "Electricity_Usage_decimal_scaled")
        st.markdown(generate_conclusion(stats), unsafe_allow_html=True)
    with st.expander("🔎 查看數據"):
        st.dataframe(transformed_df[["Electricity_Usage", "Electricity_Usage_decimal_scaled"]].style.format('{:.4f}'))

# --- Avg_Temperature ---
st.subheader("🌡️ 月均溫 (Average Temperature/Monthly)")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("##### 極值正規化 (Min-Max)")
    fig = plot_distribution(transformed_df, "Avg_Temperature_min_max", "#F5C65D", "Min-Max Normalization")
    st.pyplot(fig)
    with st.expander("📊 結論：極值正規化 (Min-Max)"):
        st.markdown("""
        **方法定義**：將資料線性縮放到 <span style='color:#4481D7'>**[0, 1]**</span> 範圍內，對異常值較為敏感，因為最大值和最小值會影響整體縮放結果
        """, unsafe_allow_html=True)
        st.markdown("--- ")
        st.markdown("##### **數據特性**")
        stats = get_df_statistics(transformed_df, "Avg_Temperature_min_max")
        st.markdown(generate_conclusion(stats), unsafe_allow_html=True)
    with st.expander("🔎 查看數據"):
        st.dataframe(transformed_df[["Avg_Temperature", "Avg_Temperature_min_max"]].style.format('{:.4f}'))

with col2:
    st.markdown("##### Z分配標準化 (Z-score)")
    fig = plot_distribution(transformed_df, "Avg_Temperature_z_score", "#45C8C5", "Z-score Normalization")
    st.pyplot(fig)
    with st.expander("📊 結論：Z分配標準化 (Z-score)"):
        st.markdown("""
        **方法定義**：將資料轉換為平均值為 <span style='color:#4481D7'>**0**</span>、標準差為 <span style='color:#4481D7'>**1**</span> 的分佈。適用於需要比較不同尺度特徵或期望資料呈常態分佈的演算法
        """, unsafe_allow_html=True)
        st.markdown("--- ")
        st.markdown("##### **數據特性**")
        stats = get_df_statistics(transformed_df, "Avg_Temperature_z_score")
        st.markdown(generate_conclusion(stats), unsafe_allow_html=True)
    with st.expander("🔎 查看數據"):
        st.dataframe(transformed_df[["Avg_Temperature", "Avg_Temperature_z_score"]].style.format('{:.4f}'))

with col3:
    st.markdown("##### 十進位正規化 (Decimal Scaling)")
    fig = plot_distribution(transformed_df, "Avg_Temperature_decimal_scaled", "#DD6D6A", "Decimal Scaling Normalization")
    st.pyplot(fig)
    with st.expander("📊 結論：十進位正規化 (Decimal Scaling)"):
        st.markdown("##### **方法定義**")
        st.markdown("""
                    透過移動小數點來實現資料縮放，使其絕對值小於 <span style='color:#4481D7'><b>1</b></span>，縮放因子取決於資料的最大絕對值，是一種簡單的正規化方法
                    """, unsafe_allow_html=True
                    )
        st.markdown("--- ")
        st.markdown("##### **數據特性**")
        stats = get_df_statistics(transformed_df, "Avg_Temperature_decimal_scaled")
        st.markdown(generate_conclusion(stats), unsafe_allow_html=True)
    with st.expander("🔎 查看數據"):
        st.dataframe(transformed_df[["Avg_Temperature", "Avg_Temperature_decimal_scaled"]].style.format('{:.4f}'))

st.divider()

# --- Discretization Section ---
st.header("⚽ 離散化 (Discretization)")
st.markdown("""
<div class="info-box">
<ul>
    <li>離散化是將連續數值型資料轉換為有限、離散的區間（或稱為「裝箱」）的過程，有助於簡化資料、減少雜訊並提高模型性能</li>
    <li>本區塊將示範不同的離散化方法及其對資料分佈的影響</li>
</ul>
</div>
""", unsafe_allow_html=True)

selected_method = st.radio(
    "選擇分析方法：",
    options=["等寬裝箱法 (Equal-width Binning)", "等深裝箱法 (Equal-depth Binning)", "決策樹分析 (Decision Tree analysis)", "聚類分析 (Clustering analysis)", "相關性分析 (Chi-Squared Analysis)"],
    key='discretize_method'
)

if "相關性分析" in selected_method:
    st.subheader("卡方獨立性檢定設定")
    chi2_col1, chi2_col2 = st.columns(2)
    with chi2_col1:
        feature1 = st.selectbox("選擇特徵一：", options=['Avg_Temperature', 'Electricity_Usage'], key='chi2_feat1')
        bins1 = st.slider("特徵一的裝箱數量：", 2, 10, 5, key='chi2_bins1')
    with chi2_col2:
        feature2 = st.selectbox("選擇特徵二：", options=['Electricity_Usage', 'Avg_Temperature'], index=1, key='chi2_feat2')
        bins2 = st.slider("特徵二的裝箱數量：", 2, 10, 5, key='chi2_bins2')

    if st.button("執行卡方檢定", key='run_chi2'):
        if feature1 == feature2:
            st.error("請選擇兩個不同的特徵進行分析。" )
        else:
            try:
                binned_feat1 = preprocessor.apply_equal_depth_binning(feature1, bins=bins1)
                binned_feat2 = preprocessor.apply_equal_depth_binning(feature2, bins=bins2)
                contingency_table = pd.crosstab(binned_feat1, binned_feat2)
                chi2, p, dof, expected = chi2_contingency(contingency_table)
                st.markdown(f"#### **{feature1} vs. {feature2} 卡方檢定結果**")
                with st.expander("🔎 查看列聯表 (Contingency Table)"):
                    st.dataframe(contingency_table)
                st.markdown("##### **檢定統計量**")
                st.markdown(generate_chi2_conclusion(chi2, p, dof), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"執行卡方檢定時發生錯誤：{e}")

elif "決策樹分析" in selected_method:
    st.subheader("決策樹裝箱設定")
    dt_col1, dt_col2 = st.columns(2)
    with dt_col1:
        selected_column_for_discretization = st.selectbox(
            "選擇要離散化的特徵：",
            options=['Avg_Temperature'], # Electricity_Usage is the target
            key='discretize_column_dt'
        )
    with dt_col2:
        max_depth = st.slider(
            "選擇決策樹最大深度 (Max Depth)：",
            min_value=2,
            max_value=5,
            value=3,
            step=1,
            key='max_depth_dt',
            help="決策樹的深度將影響最終的裝箱數量。深度為 N 最多可能產生 2^N 個裝箱"
        )
    
    if selected_column_for_discretization:
        try:
            method_name = "決策樹裝箱"
            discretized_series = preprocessor.apply_decision_tree_binning(
                feature_col=selected_column_for_discretization, 
                max_depth=max_depth
            )
            method_definition = "決策樹裝箱是一種監督式方法，它會根據目標變數（此處為用電量）來找出特徵的最佳分割點，以最大化區間之間的資訊純度"

            discretized_df = pd.DataFrame({
                '原始值': original_df[selected_column_for_discretization],
                '離散化區間': discretized_series
            })

            st.markdown(f"#### **{selected_column_for_discretization} - {method_name}結果**")
            fig_discretized = plot_discretized_data(discretized_series, selected_column_for_discretization, '#765734', f'{selected_column_for_discretization} {method_name}分佈')
            st.pyplot(fig_discretized)

            with st.expander(f"📊 結論"):
                st.markdown("##### **方法定義**")
                st.markdown(f"{method_definition}", unsafe_allow_html=True)
                st.markdown("--- ")
                st.markdown("##### **數據特性**")
                st.markdown(generate_discretization_conclusion(discretized_series, method_name, selected_column_for_discretization), unsafe_allow_html=True)
            
            with st.expander("🔎 查看離散化數據"):
                st.dataframe(discretized_df.style.format({"原始值": "{:.4f}"}))

        except (ValueError, TypeError, RuntimeError) as e:
            st.error(f"離散化錯誤：{e}")

elif "聚類分析" in selected_method:
    st.subheader("聚類分析裝箱設定")
    cluster_col1, cluster_col2 = st.columns(2)
    with cluster_col1:
        selected_column_for_clustering = st.selectbox(
            "選擇要離散化的特徵：",
            options=['Electricity_Usage', 'Avg_Temperature'],
            key='discretize_column_cluster'
        )
    with cluster_col2:
        n_clusters = st.slider(
            "選擇聚類數量 (Number of Clusters)：",
            min_value=2,
            max_value=10,
            value=4,
            step=1,
            key='n_clusters_cluster'
        )
    
    if selected_column_for_clustering:
        try:
            method_name = "K-Means 聚類裝箱"
            discretized_series = preprocessor.apply_clustering_binning(
                feature_col=selected_column_for_clustering, 
                n_clusters=n_clusters
            )
            method_definition = "K-Means 聚類裝箱是一種「非監督式」方法，它將特徵值分組成 K 個群組，使得同群組內的資料點相似度最高"

            discretized_df = pd.DataFrame({
                '原始值': original_df[selected_column_for_clustering],
                '離散化區間': discretized_series,
                '用電量': original_df['Electricity_Usage']
            })

            st.markdown(f"#### **{selected_column_for_clustering} - {method_name}結果**")
            
            res_col1, res_col2 = st.columns(2)
            category_order = discretized_series.cat.categories
            with res_col1:
                st.markdown("##### **各區間資料點計數**")
                fig_discretized = plot_discretized_data(discretized_series, selected_column_for_clustering, '#765734', f'{selected_column_for_clustering} {method_name}分佈', order=category_order)
                st.pyplot(fig_discretized)

            with res_col2:
                st.markdown("##### **各區間用電量分佈 (事後驗證)**")
                fig_validation = plot_clustering_validation(discretized_df, '離散化區間', '用電量', '#9FCE63', order=category_order)
                st.pyplot(fig_validation)
            
            with st.expander("📊 結論"):
                st.markdown("##### **方法定義**")
                st.markdown(f"{method_definition}", unsafe_allow_html=True)
                st.markdown("--- ")
                st.markdown("##### **數據特性**")
                st.markdown(generate_discretization_conclusion(discretized_series, method_name, selected_column_for_clustering), unsafe_allow_html=True)
                st.markdown("--- ")
                st.markdown("##### **與目標關聯性**")
                st.markdown(generate_clustering_conclusion(discretized_df, '離散化區間', '用電量'), unsafe_allow_html=True)

            with st.expander("🔎 查看離散化數據"):
                st.dataframe(discretized_df[["原始值", "離散化區間"]].style.format({"原始值": "{:.4f}"}))

        except (ValueError, TypeError, RuntimeError) as e:
            st.error(f"離散化錯誤：{e}")

else: # Binning methods
    st.subheader("裝箱設定")
    bin_col1, bin_col2 = st.columns(2)
    with bin_col1:
        selected_column_for_discretization = st.selectbox(
            "選擇要離散化的特徵：",
            options=['Electricity_Usage', 'Avg_Temperature'],
            key='discretize_column'
        )
    with bin_col2:
        num_bins = st.slider(
            "選擇裝箱數量 (Bins)：",
            min_value=2,
            max_value=10,
            value=5,
            step=1,
            key='num_bins_discretize'
        )

    if selected_column_for_discretization:
        try:
            if "等寬" in selected_method:
                method_name = "等寬裝箱"
                discretized_series = preprocessor.apply_equal_width_binning(selected_column_for_discretization, bins=num_bins)
                method_definition = "等寬裝箱法將資料的最小值到最大值之間的範圍劃分成相等寬度的區間，但不保證每個區間內的資料點數量會是相等"
            elif "等深" in selected_method:
                method_name = "等深裝箱"
                discretized_series = preprocessor.apply_equal_depth_binning(selected_column_for_discretization, bins=num_bins)
                method_definition = "等深裝箱法會將資料排序後，盡量將相同數量的資料點分配到每個區間中，但每個區間的寬度可能會不同"
            
            discretized_df = pd.DataFrame({
                '原始值': original_df[selected_column_for_discretization],
                '離散化區間': discretized_series
            })

            st.markdown(f"#### **{selected_column_for_discretization} - {method_name}結果**")
            fig_discretized = plot_discretized_data(discretized_series, selected_column_for_discretization, '#765734', f'{selected_column_for_discretization} {method_name}分佈')
            st.pyplot(fig_discretized)

            with st.expander("📊 結論"):
                st.markdown("##### **方法定義**")
                st.markdown(f"{method_definition}", unsafe_allow_html=True)
                st.markdown("--- ")
                st.markdown("##### **數據特性**")
                st.markdown(generate_discretization_conclusion(discretized_series, method_name, selected_column_for_discretization), unsafe_allow_html=True)
            
            with st.expander("🔎 查看離散化數據"):
                st.dataframe(discretized_df.style.format({"原始值": "{:.4f}"}))

        except (ValueError, TypeError) as e:
            st.error(f"離散化錯誤：{e}")

st.divider()

# --- Smoothing Section ---
st.header("🎾 資料平滑化 (Data Smoothing)")
st.markdown("""
<div class="info-box">
<ul>
    <li>資料平滑化旨在消除資料中的短期波動（雜訊），以突顯長期的趨勢或模式</li>
    <li>本區塊將使用不同的分箱方法來平滑資料，並比較其與原始資料的差異</li>
</ul>
</div>
""", unsafe_allow_html=True)

smooth_col1, smooth_col2 = st.columns([1, 2])

with smooth_col1:
    selected_smoothing_method = st.radio(
        "選擇平滑化方法：",
        options=["依分箱平均值 (By Bin Means)", "依分箱中位數 (By Bin Median)", "依分箱邊界 (By Bin Boundaries)"],
        key='smoothing_method'
    )
    selected_smoothing_column = st.selectbox(
        "選擇要平滑化的特徵：",
        options=['Electricity_Usage', 'Avg_Temperature'],
        key='smoothing_column'
    )
    smoothing_bins = st.slider(
        "選擇分箱數量 (Bins)：",
        min_value=2,
        max_value=20,
        value=10,
        step=1,
        key='smoothing_bins'
    )

if selected_smoothing_column:
    try:
        original_series = original_df[selected_smoothing_column]
        
        if "平均值" in selected_smoothing_method:
            method_name = "依分箱平均值"
            smoothed_series = preprocessor.smooth_by_bin_mean(selected_smoothing_column, bins=smoothing_bins)
        elif "中位數" in selected_smoothing_method:
            method_name = "依分箱中位數"
            smoothed_series = preprocessor.smooth_by_bin_median(selected_smoothing_column, bins=smoothing_bins)
        else: # Boundaries
            method_name = "依分箱邊界"
            smoothed_series = preprocessor.smooth_by_bin_boundaries(selected_smoothing_column, bins=smoothing_bins)

        with smooth_col2:
            st.markdown(f"#### **{selected_smoothing_column} - {method_name} 結果**")
            fig_smoothing = plot_smoothing_results(original_series, smoothed_series, 
                                                   f"{selected_smoothing_column} - {method_name} vs. 原始資料",
                                                   original_color='#3CBBDE', smoothed_color='#DD6D6A')
            st.pyplot(fig_smoothing)

        with st.expander("📊 結論"):
            st.markdown(generate_smoothing_conclusion(method_name, selected_smoothing_column, smoothing_bins), unsafe_allow_html=True)

        with st.expander("🔎 查看平滑化數據"):
            smoothed_df = pd.DataFrame({
                '原始值': original_series,
                '平滑後的值': smoothed_series
            })
            st.dataframe(smoothed_df.style.format('{:.4f}'))

    except (ValueError, TypeError) as e:
        st.error(f"平滑化處理時發生錯誤：{e}")