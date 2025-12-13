import streamlit as st
import base64
import pandas as pd
import textwrap
from src.data_preprocessing import DataPreprocessor
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

st.set_page_config(page_title="資料分析", page_icon="📊", layout="wide")

st.title("📊 資料整合分析 (Data Integration Analysis)")

# Helper function to generate conclusions for plots
def generate_analysis_conclusion(plot_type, analysis_data, encoders=None):
    conclusion_text = ""
    if plot_type == 'linear_regression':
        stats = analysis_data['linear_regression_stats']
        slope = stats['slope']
        r_squared = stats['r_squared']
        p_value = stats['p_value']

        if p_value < 0.05:
            sig_text = "統計結果顯著"
        else:
            sig_text = "統計結果不顯著"

        conclusion_text += f"- **相關性強度**：此模型 R-squared (R²) = <span style='color:#4481D7'>**{r_squared:.4f}**</span> 表示此模型解釋用電量約 <span style='color:#4481D7'>**{r_squared*100:.4f}%**</span> 的變異，即月均溫與用電量之間存在中等至強的線性關係\n"
        conclusion_text += f"- **關係方向**：此模型斜率 (Slope｜Coefficient) = <span style='color:#4481D7'>**{slope:.4f}**</span>，表示月均溫每增加 1 單位，用電量平均增加約 <span style='color:#4481D7'>**{slope:.4f}**</span> 萬KW\n"
        if slope > 0:
            conclusion_text += "- **趨勢**：此模型呈現<span style='color:#4481D7'>**正向關係**</span>，即月均溫升高時，用電量傾向於增加\n"
        else:
            conclusion_text += "- **趨勢**：此模型呈現<span style='color:#4481D7'>**負向關係**</span>，即月均溫升高時，用電量傾向於減少\n"
        
        if p_value < 0.01:
            conclusion_text += f"- **統計結果**：此模型 p-value = <span style='color:#4481D7'>**{p_value:.4f}**</span> ，有強烈證據支持月均溫與用電量之間存在線性關係\n"
        elif p_value < 0.05:
            conclusion_text += f"- **統計結果**：此模型 p-value = <span style='color:#4481D7'>**{p_value:.4f}**</span>，有適度證據支持月均溫與用電量之間存在線性關係\n"
        else:
            conclusion_text += f"- **統計結果**：此模型 p-value = <span style='color:#4481D7'>**{p_value:.4f}**</span>，無足夠證據支持月均溫與用電量之間存在線性關係\n"

    elif plot_type == 'residual_plot':
        st.markdown("##### **殘差圖觀察重點**")
        st.markdown("- <span style='color:#4481D7'>**零點水平線**</span>：理想的殘差應該圍繞此線隨機分佈", unsafe_allow_html=True)
        st.markdown("- <span style='color:#4481D7'>**非線性關係**</span>：若殘差呈現曲線形態，可能暗示變數間存在非線性關係，可考慮更複雜的模型或資料轉換", unsafe_allow_html=True)
        st.markdown("- <span style='color:#4481D7'>**異方差性**</span>：若殘差呈現喇叭形或漏斗形分佈，可能暗示誤差變異數非恆定，可考慮更複雜的模型或資料轉換", unsafe_allow_html=True)

    elif plot_type == 'boxplot':
        median_data = analysis_data
        col_name = median_data.name
        
        if encoders and col_name in encoders:
            original_labels = {i: label for i, label in enumerate(encoders[col_name])}
            median_data.index = median_data.index.map(original_labels)

        if not median_data.empty:
            highest_median_cat = median_data.index[0]
            highest_median_val = median_data.iloc[0]
            lowest_median_cat = median_data.index[-1]
            lowest_median_val = median_data.iloc[-1]
            median_range = highest_median_val - lowest_median_val

            st.markdown(f"##### **{col_name} 用電量中位數排序：**")
            median_df = pd.DataFrame({'類別': median_data.index, '用電量中位數（萬KW）': median_data.values})
            st.dataframe(median_df.style.format({'用電量中位數（萬KW）': '{:.4f}'}))
            conclusion_text += f"- **中位數範圍**：各類別用電量中位數差異為 <span style='color:#4481D7'>**{median_range:.4f}**</span> 萬KW，顯示該類別對用電量有不同程度的影響"
        else:
            conclusion_text += "無法計算此類別的結論"
    return conclusion_text

# Function to generate plots, cached
@st.cache_data
def generate_integration_plots(_preprocessor):
    try:
        return _preprocessor.integrate_data()
    except Exception as e:
        st.error(f"生成圖表時發生錯誤：{e}")
        return None, None

def generate_rules_summary(sorted_rules_df):
    """Generates a human-readable summary of the top association rules from a pre-sorted DataFrame."""
    if sorted_rules_df.empty:
        return ""

    summary = "#### 🚨 分析摘要 (Analysis Summary)\n\n"
    top_rules = sorted_rules_df.head(3)

    for index, rule in top_rules.iterrows():
        # Correctly format antecedents and consequents with highlighting
        antecedents = ', '.join([f"<span style='color:#9ACD32'>**{item.replace('=', '為')}**</span>" for item in rule['antecedents']])
        consequents = ', '.join([f"<span style='color:#9ACD32'>**{item.replace('=', '為')}**</span>" for item in rule['consequents']])
        lift = rule['lift']
        confidence = rule['confidence']

        summary += f"**規則 {index}：**\n"
        summary += f"> 當 {antecedents} 時，可以發現 {consequents} 的可能性也隨之提高。\n"
        
        if lift > 1.5:
            lift_desc = f"此規則的增益 (Lift) 值高達 <span style='color:#9ACD32'>**{lift:.4f}**</span>，這代表一個**非常強**的正相關，意謂著這兩件事同時發生的機率遠高於偶然。"
        else:
            lift_desc = f"此規則的增益 (Lift) 值為 <span style='color:#9ACD32'>**{lift:.4f}**</span>，呈現正相關。"
        
        # Clean up the HTML string for confidence description
        conf_antecedents = antecedents.replace("<span style='color:#9ACD32'>**", "").replace("**</span>", "")
        conf_consequents = consequents.replace("<span style='color:#9ACD32'>**", "").replace("**</span>", "")
        conf_desc = f"其信賴度 (Confidence) 為 <span style='color:#9ACD32'>**{confidence:.4%}**</span>，表示在滿足「{conf_antecedents}」這個條件的資料中，有 <span style='color:#9ACD32'>**{confidence:.4%}**</span> 的情況也會滿足「{conf_consequents}」。\n\n"
        
        summary += f"{lift_desc} {conf_desc}"

    return summary


# Check for data availability
if 'cleaned_df' not in st.session_state or 'preprocessor' not in st.session_state:
    st.warning("⬅️ 請先至「📄 資料探索與清理」頁面上傳並清理資料")
    st.stop()

st.info("此頁面提供多種資料分析方法，請在下方選擇分頁進行探索。", icon="ℹ️")

# Get preprocessor and original cleaned df
preprocessor = st.session_state['preprocessor']
cleaned_df = st.session_state['cleaned_df']

# Create tabs
tab1, tab2 = st.tabs(["📊 探索式資料分析 (Exploratory Data Analysis)", "🧺 關聯規則分析 (Association Rule Mining)"])

with tab1:
    plots, analysis_results = generate_integration_plots(preprocessor)
    if plots and analysis_results:
        st.markdown("<h3>💠 特徵相關性分析（Correlation Analysis）</h3>", unsafe_allow_html=True)
        st.markdown("<h4>🌡️ 相關性熱圖 （Annotated Heatmap）</h4>", unsafe_allow_html=True)
        st.markdown(textwrap.dedent("""
                    相關性分析用於衡量兩個或多個變數之間的統計關係強度與方向
                    - 顏色愈接近 <span style='color:#4481D7'>**1**</span> 或
                      <span style='color:#4481D7'>**-1**</span>，
                      表示變數之間的線性關係愈<span style='color:#4481D7'>**強**</span>
                    - 顏色愈接近 <span style='color:#4481D7'>**0**</span>，
                      表示變數之間的線性關係愈<span style='color:#4481D7'>**弱**</span>
                    """), unsafe_allow_html=True)
        st.image(f"data:image/png;base64,{plots['correlation_heatmap']}", caption="特徵相關性熱圖")
        
        with st.expander("📊 結論：特徵相關性分析"):
            corr_matrix = analysis_results['correlation_matrix']
            if corr_matrix is not None:
                corr_with_target = corr_matrix['Electricity_Usage'].sort_values(ascending=False).drop('Electricity_Usage')
                
                st.markdown("##### **所有特徵與用電量之相關係數排序：**")
                st.dataframe(corr_with_target.reset_index().rename(columns={'index': '特徵', 'Electricity_Usage': '相關係數'}).style.format({'相關係數': '{:.4f}'}))
                
                st.markdown("從上方的熱圖與相關係數表，可以看出與「用電量」最相關的幾個變數：")
                
                top_positive = corr_with_target.nlargest(3, keep='all')
                st.markdown("##### **正相關** (數值愈大，用電量可能愈高)：")
                for feature, corr_value in top_positive.items():
                    st.markdown(f"- <span style='color:#4481D7'>**{feature}**</span>：相關係數為 <span style='color:#4481D7'>**{corr_value:.4f}**</span>", unsafe_allow_html=True)
                
                top_negative = corr_with_target.nsmallest(3, keep='all').sort_values()
                st.markdown("##### **負相關** (數值愈大，用電量可能愈低)：")
                for feature, corr_value in top_negative.items():
                     st.markdown(f"- <span style='color:#4481D7'>**{feature}**</span>：相關係數為 <span style='color:#4481D7'>**{corr_value:.4f}**</span>", unsafe_allow_html=True)
            else:
                st.markdown("無法計算相關性矩陣。")

        st.divider()
        st.markdown("<h3>🔢 數值型變數與用電量關係 (Numerical Variables & Electricity Usage)</h3>", unsafe_allow_html=True)
        st.markdown("<h4>📈 線性迴歸圖（Linear Regression Plot）</h4>", unsafe_allow_html=True)
        st.markdown("顯示兩個數值型變數之間的線性關係，並包含迴歸線與邊際分佈圖")
        if 'jointplot_reg' in plots:
            st.image(f"data:image/png;base64,{plots['jointplot_reg']}", caption="線性迴歸暨邊際分佈圖")
        
        with st.expander("📊 結論：線性迴歸分析"):
            st.markdown(textwrap.dedent("""
            ##### 方法定義
            顯示兩個數值型變數之間的線性關係，並包含迴歸線與邊際分佈圖
            - **迴歸線**：「紅色實線」表示資料的最佳擬合線；「紅色實線的斜率」表示兩個變數之間的關係：
                - <span style='color:#4481D7'>**正斜率**</span>：表示一個變數增加時，另一個變數也傾向於增加
                - <span style='color:#4481D7'>**負斜率**</span>：表示一個變數增加時，另一個變數卻傾向於減少
            - **散佈點**：「藍色點」表示個別資料點
                - 資料點愈接近 <span style='color:#4481D7'>**迴歸線**</span>，表示線性關係愈<span style='color:#4481D7'>**強**</span>
            - **邊際分佈圖**：上方和右側的直方圖顯示單獨變數的分佈情況
            """), unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("##### **數據特性**")
            st.markdown(generate_analysis_conclusion('linear_regression', analysis_results), unsafe_allow_html=True)

        st.markdown("<h4>📉 殘差圖（Residual Plot）</h4>", unsafe_allow_html=True)
        st.markdown(textwrap.dedent("""
                    殘差圖顯示「觀測的預測值」與「觀測的殘差」之間的關係，
                    有助於評估迴歸模型的適用性並檢查是否存在非線性關係或異方差性
                    - 觀測的殘差：預測回應值與實際回應值之間的差異
                    """))
        st.image(f"data:image/png;base64,{plots['residual_temp_vs_elec']}", caption="月均溫與用電量殘差圖")

        with st.expander("📊 結論：殘差分析"):
            st.markdown(generate_analysis_conclusion('residual_plot', None), unsafe_allow_html=True)

        st.divider()
        st.markdown("<h3>🔖 類別型變數與用電量關係 (Categorical Variables & Electricity Usage)</h3>", unsafe_allow_html=True)
        st.markdown("<h4>📦 盒鬚圖（Box Plot）</h4>", unsafe_allow_html=True)
        st.markdown(textwrap.dedent("""
                    盒鬚圖是一種標準化的方式，用於顯示資料的分佈情況、集中趨勢和變異數
                    - 可以比較不同類別變數在用電量上的分佈差異
                    - 它將資料分為「四分位數」，能夠清楚地展示資料的離散程度、偏態和異常值
                    """))
        
        categorical_cols = ['Science_Park', 'Sub_Science_Park', 'County', 'Town']
        col1, col2 = st.columns(2)
        
        for i, col in enumerate(categorical_cols):
            target_col = col1 if i % 2 == 0 else col2
            with target_col:
                plot_key = f'boxplot_{col}_vs_elec'
                if plot_key in plots:
                    st.image(f"data:image/png;base64,{plots[plot_key]}", caption=f"{col} 與用電量盒鬚圖")
                    with st.expander(f"📊 {col} 結論"):
                        st.markdown(textwrap.dedent(f"""
                        ##### 方法定義
                        透過五數綜述（最小值、第一四分位數、中位數、第三四分位數、最大值）來展示 <span style='color:#4481D7'>**{col}**</span> 的用電量分佈、集中趨勢和離散程度
                        """), unsafe_allow_html=True)
                        st.markdown("---")
                        st.markdown("##### **數據特性**")
                        median_data = analysis_results.get('box_plot_analysis', {}).get(col)
                        if median_data is not None and not median_data.empty:
                            st.markdown(generate_analysis_conclusion('boxplot', median_data, preprocessor.encoders), unsafe_allow_html=True)
                        else:
                            st.markdown("無法計算此類別的結論。", unsafe_allow_html=True)
    else:
        st.error("無法生成或載入分析圖表。請確認資料是否正確。")

with tab2:
    st.header("🛒 關聯規則分析 (Association Rule Mining)")
    st.markdown(textwrap.dedent("""
        - 常用於「市場購物籃分析 (Market Basket Analysis)」，旨在發掘資料集中項目之間的有趣關係
        - 將資料的每個 row 視為一筆交易，每個特徵的數值視為交易中的一個「項目」，藉此找出特徵之間的 `if-then` 關聯規則
    """), unsafe_allow_html=True)

    with st.container(border=True):
        st.subheader("⚙️ 分析設定 (Analysis Settings)")
        
        all_cols = ['Science_Park', 'Sub_Science_Park', 'County', 'Town', 'Year_EN', 'Month_NUM', 'Avg_Temperature', 'Electricity_Usage']
        
        selected_features = st.multiselect(
            '**1. 選擇要分析的特徵 (Select Features for Analysis)**',
            options=all_cols,
            default=['Sub_Science_Park', 'Month_NUM', 'Avg_Temperature', 'Electricity_Usage'],
            help="選擇您感興趣的特徵來進行關聯規則分析。建議不要選擇過多特徵，以免規則過於複雜。"
        )

        st.markdown("**2. 設定連續型資料的離散化區間數 (Set Bins for Continuous Features)**")
        st.caption("⚠️ 此步驟是將連續數值（如溫度、用電量）轉換為類別（如高、中、低），這是 Apriori 演算法的必要前置處理。預設值 `3` 代表將數值分為「低、中、高」三個等級")
        
        preprocessor = st.session_state['preprocessor']
        continuous_cols = [
            col for col in selected_features 
            if col not in preprocessor.encoders 
            and cleaned_df[col].dtype in ['float64', 'int64'] 
            and cleaned_df[col].nunique() > 10
        ]
        
        bins_config = {}
        if continuous_cols:
            col1, col2, col3 = st.columns(3)
            cols = [col1, col2, col3]
            for i, col in enumerate(continuous_cols):
                with cols[i % 3]:
                    bins_config[col] = st.number_input(f"`{col}` 的區間數", min_value=2, max_value=10, value=3, key=f"bins_{col}")
        else:
            st.info("您選擇的特徵中沒有需要離散化的連續型資料。")

        st.markdown("**3. 設定 Apriori 演算法參數 (Set Apriori Parameters)**")
        col1, col2 = st.columns(2)
        with col1:
            min_support = st.slider('最低支持度 (Min Support)', 0.01, 1.0, 0.05, 0.01, help="一個項目集在所有交易中出現的頻率。較高的值會篩選掉不常見的項目集。")
        with col2:
            min_confidence = st.slider('最小信賴度 (Min Confidence)', 0.1, 1.0, 0.5, 0.1, help="規則的可靠性指標。`IF {A} THEN {B}` 的信賴度是指交易中包含 A 時，也包含 B 的機率。")

        run_button = st.button('🚀 執行關聯規則分析', type="primary", use_container_width=True)

    if 'rules_df' not in st.session_state:
        st.session_state.rules_df = pd.DataFrame()

    if run_button:
        if not selected_features:
            st.warning("請至少選擇一個特徵進行分析。")
        else:
            with st.spinner('正在進行分析，請稍候...'):
                try:
                    df_apriori = cleaned_df[selected_features].copy()

                    # Inverse transform categorical data to get original labels
                    preprocessor = st.session_state['preprocessor']
                    for col in df_apriori.columns:
                        if col in preprocessor.encoders:
                            # The encoder is a list of categories. The data is the integer code.
                            # We map the integer code back to the string category.
                            # df.cat.codes uses -1 for NaN, so we handle that.
                            category_list = preprocessor.encoders[col]
                            df_apriori[col] = df_apriori[col].astype(int).apply(lambda x: category_list[x] if x >= 0 and x < len(category_list) else 'N/A')

                    bin_labels = {2: ['Low', 'High'], 3: ['Low', 'Medium', 'High'], 4: ['Lowest', 'Low', 'High', 'Highest']}
                    for col, bins in bins_config.items():
                        labels = bin_labels.get(bins, [f"Bin_{i}" for i in range(bins)])
                        df_apriori[col] = pd.qcut(df_apriori[col], q=bins, labels=labels, duplicates='drop')

                    transactions = [
                        [f"{col}={df_apriori[col].iloc[i]}" for col in df_apriori.columns]
                        for i in range(len(df_apriori))
                    ]
                    
                    te = TransactionEncoder()
                    te_ary = te.fit(transactions).transform(transactions)
                    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

                    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

                    if frequent_itemsets.empty:
                        st.warning("在此支持度設定下，找不到任何高頻項目集。請嘗試調低「最低支持度」。")
                        st.session_state.rules_df = pd.DataFrame()
                    else:
                        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
                        if rules.empty:
                            st.warning("雖然找到了高頻項目集，但在目前的信賴度設定下，無法生成任何關聯規則。請嘗試調低「最小信賴度」。")
                            st.session_state.rules_df = pd.DataFrame()
                        else:
                            st.success(f"分析完成！共找到 {len(rules)} 條關聯規則。")
                            st.session_state.rules_df = rules.sort_values(by='lift', ascending=False).reset_index(drop=True)

                except Exception as e:
                    st.error(f"分析過程中發生錯誤：{e}")
                    st.session_state.rules_df = pd.DataFrame()

    if not st.session_state.rules_df.empty:
        st.subheader("⛳ 關聯規則結果 (Association Rules)")
        st.dataframe(st.session_state.rules_df)

        st.markdown("---")
        summary = generate_rules_summary(st.session_state.rules_df)
        st.markdown(summary, unsafe_allow_html=True)
        
        with st.expander("📖 如何解讀關聯規則？"):
            st.markdown("""
            - **antecedents (前項)**：規則的 "IF" 部分
            - **consequents (後項)**：規則的 "THEN" 部分
            - **support (支持度)**：規則中「前項和後項一起出現」的交易比例，反映規則在整體資料中的普遍性
            - **confidence (信賴度)**：在包含「前項」的交易中，同時也包含「後項」的比例，即衡量規則的準確性
            - **lift (增益)**：衡量「後項」在給定「前項」的情況下，其出現機率相對於其自身獨立出現機率的提升程度
                - **Lift > 1**：表示前項和後項存在**正相關**，前項的發生，會提升後項發生的機率
                - **Lift < 1**：表示前項和後項存在**負相關**，前項的發生，會降低後項發生的機率
                - **Lift = 1**：表示前項和後項**相互獨立**，沒有關聯
            
            ---
            ##### **進階指標**
            - **representativity (代表性)**：衡量規則在整體資料中的代表性，計算方式為 `support(A ∪ B) / support(B)`
                - 值接近 1：表示規則對後項具有高度代表性
                - 值接近 0：表示規則對後項的代表性較低
            - **leverage (槓桿率)**：量測前項與後項同時出現的頻率，比「假設兩者獨立時的預期頻率」高出多少
                - 值為 0：表示獨立
                - 大於 0：表示同時出現的頻率高於預期
            - **conviction (確信度)**：用來衡量「前項」對於「後項」的影響力，一個高確信度值意味著後項的發生高度依賴於前項
                - 例如：若 `conviction` 為 2，表示如果規則沒有後項，它的出錯率會是原來的 2 倍
            - **zhangs_metric (張氏指標)**：一個綜合指標，範圍在 -1 到 +1 之間
                - 值接近 +1：表示強正相關
                - 值接近 -1：表示強負相關
                - 值接近 0：表示無關聯
            - **jaccard (雅卡爾指數)**：衡量前項和後項同時出現的頻率與至少有一個出現的頻率之比
                - 值介於 0 到 1 之間，值愈大表示兩者關聯愈強
            - **certainty (確定性)**：衡量在前項發生的情況下，後項發生的增強程度
                - 值介於 -1 到 +1 之間
                - 正值：表示前項的發生增加了後項發生的可能性
                - 負值：表示前項的發生減少了後項發生的可能性
            - **Kulczynski (庫氏指標)**：衡量前項和後項之間的對稱關聯性
                - 值介於 0 到 1 之間，值愈大表示關聯愈強
            """, unsafe_allow_html=True)
    elif run_button: # To ensure messages are shown after a run that results in an empty dataframe
        pass # The warnings are already shown inside the `if run_button` block