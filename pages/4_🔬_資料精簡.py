import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import textwrap
import io
import base64
import pywt
from src.data_preprocessing import DataPreprocessor
from src.ui_components import render_app_info, render_data_status

# Helper function for Scree Plot
def plot_scree_plot(pca_model):
    fig, ax = plt.subplots(figsize=(10, 6))
    num_components = len(pca_model.explained_variance_ratio_)
    components = np.arange(1, num_components + 1)
    
    ax.plot(components, pca_model.explained_variance_ratio_, 'o-', linewidth=2, color='#3CBBDE', label='個別解釋變異量')
    ax.set_xlabel('主成分數量')
    ax.set_ylabel('解釋變異量百分比')
    ax.set_title('碎石圖 (Scree Plot)')
    ax.grid(True)

    # Add cumulative explained variance
    cumulative_variance = np.cumsum(pca_model.explained_variance_ratio_)
    ax.plot(components, cumulative_variance, 'x-', linewidth=2, color='#DD6D6A', label='累積解釋變異量')
    ax.legend()

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# Helper function for PCA 2D Scatter Plot
def plot_pca_2d_scatter(pca_df, original_df, target_col='Electricity_Usage'):
    if pca_df.shape[1] < 2:
        return None # Not enough components for 2D plot

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(pca_df['PC_1'], pca_df['PC_2'], c=original_df[target_col], cmap='viridis', alpha=0.7)
    ax.set_xlabel('主成分 1')
    ax.set_ylabel('主成分 2')
    ax.set_title('PCA 2D 散佈圖 (PCA 2D Scatter Plot)')
    
    cbar = plt.colorbar(scatter)
    cbar.set_label(target_col)
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# Helper function for PCA 3D Scatter Plot
def plot_pca_3d_scatter(pca_df, original_df, target_col='Electricity_Usage'):
    if pca_df.shape[1] < 3:
        return None # Not enough components for 3D plot

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(pca_df['PC_1'], pca_df['PC_2'], pca_df['PC_3'], c=original_df[target_col], cmap='viridis', alpha=0.7)
    
    ax.set_xlabel('主成分 1')
    ax.set_ylabel('主成分 2')
    ax.set_zlabel('主成分 3')
    ax.set_title('PCA 3D 散佈圖 (PCA 3D Scatter Plot)')
    
    cbar = plt.colorbar(scatter, pad=0.1)
    cbar.set_label(target_col)
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# Helper function to generate PCA conclusion
def generate_pca_conclusion(pca_model, n_components):
    explained_variance_ratio = pca_model.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    st.markdown("##### 方法定義")
    st.markdown("""
                - **主成分分析 (PCA)**：一種線性降維技術，透過正交轉換將原始特徵轉換為一組新的不相關特徵（主成分），再按其解釋資料變異量作排序
  - 若累積解釋變異量已達高比例 (如 80%-90% 以上)，則所選主成分能有效代表原始資料
  - 碎石圖上「手肘」處（斜率顯著變緩<span style='color:#4481D7'>**前**</span>）的主成分數量通常是較佳的選擇
                """, unsafe_allow_html=True)
    
    st.divider()

    st.markdown("##### 數據特性")
    conclusion = f"""
- **選擇主成分數量**：選擇 <span style='color:#4481D7'>**{n_components}**</span> 個主成分
- **解釋變異量**：
"""
    for i in range(n_components):
        conclusion += f"  - 主成分 <span style='color:#4481D7'>**{i+1}**</span> 解釋 <span style='color:#4481D7'>**{explained_variance_ratio[i]:.2%}**</span> 的變異量\n"
    conclusion += f"- **累積解釋變異量**：前 <span style='color:#4481D7'>**{n_components}**</span> 個主成分共解釋了 <span style='color:#4481D7'>**{cumulative_variance[n_components-1]:.2%}**</span> 的總變異量\n"
    
    return textwrap.dedent(conclusion)

# Helper function for t-SNE 2D Scatter Plot
def plot_tsne_2d_scatter(tsne_df, original_df, target_col='Electricity_Usage'):
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(tsne_df['t-SNE 1'], tsne_df['t-SNE 2'], c=original_df[target_col], cmap='viridis', alpha=0.7)
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_title('t-SNE 2D 散佈圖 (t-SNE 2D Scatter Plot)')
    
    cbar = plt.colorbar(scatter)
    cbar.set_label(target_col)
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# Helper function to generate t-SNE conclusion
def generate_tsne_conclusion(perplexity, learning_rate):
    
    st.markdown("##### t-SNE 參數說明")
    st.markdown("""
- **困惑度 (Perplexity)**：影響每個點的近鄰數量，通常介於 5 到 50 之間
    - <span style='color:#4481D7'>**低**</span> 的困惑度 (如：5-10) → 強調 <span style='color:#4481D7'>**局部結構**</span>，可能產生緊密但分散的小群集
    - <span style='color:#4481D7'>**高**</span> 的困惑度 (如：30-50) → 強調 <span style='color:#4481D7'>**全局結構**</span>，可能將資料點融合成一個或幾個大群集
- **學習率 (Learning Rate)**：控制點位置更新的步伐大小，常見範圍為 10 到 1000
    - <span style='color:#4481D7'>**低**</span> 的學習率 (如：10-100) → 可能導致收斂緩慢，可能卡在局部最優解，導致圖形過於擁擠
    - <span style='color:#4481D7'>**高**</span> 的學習率 (如：1000) → 可能導致不穩定的結果，導致點「跳躍」得太遠而無法收斂到一個好的解，使得圖形看起來像一個混亂的球
""", unsafe_allow_html=True)

    st.divider()

    conclusion = f"""
 🌟 建議嘗試不同的參數組合，以觀察哪種最能揭示資料中有意義的模式
- 設定困惑度 (Perplexity) = <span style='color:#4481D7'>**{perplexity}**</span>
- 設定學習率 (Learning Rate) = <span style='color:#4481D7'>**{learning_rate}**</span>

註：t-SNE 透過這些參數調整，將高維度資料點投影到二維空間以揭示資料的潛在結構和群聚情況
"""
    return textwrap.dedent(conclusion)

# Helper function to plot feature importance bar chart
def plot_feature_importance_bar_chart(feature_importances, title, color='#3CBBDE'):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=feature_importances.values, y=feature_importances.index, ax=ax, color=color)
    ax.set_title(title)
    ax.set_xlabel("重要性分數")
    ax.set_ylabel("特徵")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')



# Helper function to generate feature ranking conclusion
def generate_feature_ranking_conclusion(feature_importances, method_name):
    blue = "#4481D7"

    conclusion = f"""
- **排序方法**：使用 <span style='color:{blue}'>**{method_name}**</span> 進行特徵排序
- **最重要特徵**：
"""
    top_features = feature_importances.head(3)
    for i, (feature, score) in enumerate(top_features.items()):
        conclusion += f"  - **第 {i+1} 重要特徵**：<span style='color:{blue}'>**{feature}**</span>，重要性分數為 <span style='color:{blue}'>**{score:.4f}**</span>\n"
    
    st.divider()

    if method_name == "互資訊 (Mutual Information)":
        conclusion += f"""
- **方法定義**：
  - 互資訊衡量兩個變數之間的相互依賴程度（包含線性和非線性關係）
  - 分數愈高，表示該特徵與目標「用電量」之間的關聯性愈強，能提供更多關於目標的資訊

⚠️ **重要提醒**：互資訊僅衡量關聯性，並不表示因果關係
"""
    elif method_name == "資訊增益 (Information Gain)":
        conclusion += f"""
- **方法定義**：
  - 資訊增益衡量某個特徵的引入，能為目標「用電量」的分類減少多少不確定性
  - 分數愈高，表示該特徵對於區分「用電量」的不同等級（高、中、低）愈有幫助

⚠️ **重要提醒**：此處的資訊增益是透過將「用電量」離散化（分箱）後計算得出的
"""
    return textwrap.dedent(conclusion)

# Helper function to decode categorical values for display
def get_decoded_categorical_values(data_point, features, encoders):
    decoded_values = {}
    for feature in features:
        if feature in encoders:
            code = data_point[feature]
            # Ensure the code is a valid key in the encoder list
            if 0 <= code < len(encoders[feature]):
                decoded_values[feature] = encoders[feature][int(code)]
            else:
                decoded_values[feature] = f"無效代碼: {code}"
        else:
            decoded_values[feature] = data_point[feature] # Should not happen for categorical
    return pd.Series(decoded_values)

# Streamlit 頁面設定
st.set_page_config(page_title="資料精簡", page_icon="🔬", layout="wide")

st.title("🔬 資料精簡 (Data Reduction)")

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

# Helper function to get preprocessor and data
if 'cleaned_df' not in st.session_state or 'preprocessor' not in st.session_state:
    st.warning("⬅️ 請先至「📄 資料探索與清理」頁面上傳並清理資料" )
    st.stop()

render_app_info()
cleaned_df = st.session_state['cleaned_df']
render_data_status(cleaned_df)

original_df = st.session_state['df']
preprocessor = st.session_state['preprocessor']

st.markdown(textwrap.dedent("""
    <div class="info-box">
    <ul>
        <li>資料精簡旨在透過多種技術減少資料的複雜度，同時盡量保留其核心資訊，以提升模型效率、降低儲存成本並解決共線性問題</li>
        <li>此頁面將提供互動式介面，讓使用者探索不同的資料精簡方法及其對資料的影響</li>
    </ul>
    </div>
    """), unsafe_allow_html=True)


# Main Tabs
tab1, tab2, tab3 = st.tabs(["維度縮減 (Dimensionality Reduction)", "數量縮減 (Numerosity Reduction)", "資料壓縮 (Data Compression)"])

with tab1:
    st.header("維度縮減 (Dimensionality Reduction)")
    st.markdown(textwrap.dedent("""
        <div class="info-box">
        <ul>
            <li>維度縮減是減少資料特徵數量（欄位）的過程，同時盡量保留資料中的主要變異資訊</li>
            <li>它有助於簡化模型、減少過度擬合的風險，並提升模型的訓練速度與可解釋性</li>
        </ul>
        </div>
        """), unsafe_allow_html=True)

    # Sub-tabs for Dimensionality Reduction
    sub_tab1_1, sub_tab1_2 = st.tabs(["視覺化降維 (Visual Reduction)", "特徵排序與距離度量 (Feature Ranking & Distance Metrics)"])

    with sub_tab1_1:
        st.subheader("視覺化降維 (Visual Reduction)")
        st.markdown(textwrap.dedent("""
            <div class="info-box">
            <ul>
                <li>將高維度資料點投影到低維度空間（通常是2D），便於視覺化觀察資料點的潛在結構和群聚</li>
                <li>此處將示範 PCA 和 t-SNE 兩種常用的降維技術</li>
            </ul>
            </div>
            """), unsafe_allow_html=True)
        
        dim_reduc_method = st.radio(
            "選擇降維方法：",
            options=["主成分分析 (PCA)", "t-SNE (t-distributed Stochastic Neighbor Embedding)"],
            key="dim_reduc_method"
        )

        # Get numerical features for dimensionality reduction
        numerical_cols = cleaned_df.select_dtypes(include=np.number).columns.tolist()
        if 'Electricity_Usage' in numerical_cols:
            numerical_cols.remove('Electricity_Usage') # Usually target is not reduced with features

        if dim_reduc_method == "主成分分析 (PCA)":
            st.markdown("#### 主成分分析 (PCA)")
            st.info("PCA 旨在將資料點投影到一組新的正交特徵（主成分）上，這些主成分按其解釋資料變異量的多少排序", icon="ℹ️")

            if not numerical_cols:
                st.warning("資料中沒有足夠的數值型特徵進行 PCA 分析")
            else:
                n_components = st.slider(
                    "選擇主成分數量：",
                    min_value=1,
                    max_value=len(numerical_cols) if len(numerical_cols) <= 10 else 10, # Limit for UI demonstration
                    value=2,
                    step=1,
                    key="pca_n_components",
                    help="選擇用於降維的主成分數量。通常選擇能解釋大部分變異的少量主成分"
                )
                
                if st.button("執行 PCA"):
                    with st.spinner("執行 PCA 及繪製圖表中..."):
                        try:
                            # Execute PCA
                            pca_model, pca_df = preprocessor.apply_pca(numerical_cols, n_components)

                            col_pca_1, col_pca_2 = st.columns(2)
                            with col_pca_1:
                                # Scree Plot
                                st.markdown("##### 碎石圖 (Scree Plot)")
                                scree_plot_base64 = plot_scree_plot(pca_model)
                                st.image(f"data:image/png;base64,{scree_plot_base64}", use_container_width=True)
                            
                            with col_pca_2:
                                # Scatter Plot (2D or 3D)
                                if n_components == 2:
                                    st.markdown("##### PCA 2D 散佈圖 (PCA 2D Scatter Plot)")
                                    pca_scatter_base64 = plot_pca_2d_scatter(pca_df, cleaned_df, 'Electricity_Usage')
                                    if pca_scatter_base64:
                                        st.image(f"data:image/png;base64,{pca_scatter_base64}", use_container_width=True)
                                    else:
                                        st.warning("無法生成 2D 散佈圖，請確保主成分數量為 2")
                                elif n_components >= 3:
                                    st.markdown("##### PCA 3D 散佈圖 (PCA 3D Scatter Plot) - 顯示前三個主成分")
                                    pca_scatter_base64 = plot_pca_3d_scatter(pca_df, cleaned_df, 'Electricity_Usage')
                                    if pca_scatter_base64:
                                        st.image(f"data:image/png;base64,{pca_scatter_base64}", use_container_width=True)
                                    else:
                                        st.warning("無法生成 3D 散佈圖，請確保主成分數量至少為 3")
                                else:
                                    st.warning("若要顯示散佈圖，主成分數量必須至少為 2")

                            with st.expander("📊 結論：PCA 分析"):
                                st.markdown(generate_pca_conclusion(pca_model, n_components), unsafe_allow_html=True)
                            
                            with st.expander("🔎 查看 PCA 轉換後的數據"):
                                st.dataframe(pca_df.style.format('{:.4f}'))

                            with st.expander("🗂️ 查看主成分特徵負載 (Component Loadings)"):
                                st.info("""
                                    ℹ️ 特徵負載表示原始特徵對主成分的影響程度（「負載 (Loading)」或「權重 (Weight)」）並按絕對值作降冪
                                       
                                        ⚠️ 表格呈現方式是「以絕對值作降冪排序，但顯示原始值」
                                           絕對值大小 → 負載的絕對值愈大，代表該原始特徵對主成分的「貢獻度」或「影響力」愈多
                                           正負號 → 代表該影響是「正相關」還是「負相關」
                                    """)
                                loadings = pca_model.components_
                                loadings_df = pd.DataFrame(loadings, columns=numerical_cols, index=[f'PC_{i+1}' for i in range(n_components)])
                                
                                for i in range(n_components):
                                    st.markdown(f"##### 主成分 {i+1} 的特徵負載")
                                    pc_loadings_series = loadings_df.loc[f'PC_{i+1}']
                                    sorted_pc_loadings = pc_loadings_series.iloc[pc_loadings_series.abs().argsort()[::-1]]
                                    st.dataframe(sorted_pc_loadings.to_frame(f'PC_{i+1} Loadings').style.format('{:.4f}'))

                        except Exception as e:
                            st.error(f"執行 PCA 時發生錯誤: {e}")


        elif dim_reduc_method == "t-SNE (t-distributed Stochastic Neighbor Embedding)":
            st.markdown("#### t-SNE")
            st.info("t-SNE 是一種非線性降維技術，特別適合用於將高維度資料視覺化，即它盡力保留資料點在高維度空間中的局部結構", icon="ℹ️")
            st.warning("t-SNE 計算成本較高，且對參數敏感，建議小規模資料集或先用 PCA 觀察資料概況")
            
            if not numerical_cols:
                st.warning("資料中沒有足夠的數值型特徵進行 t-SNE 分析")
            else:
                perplexity = st.slider(
                    "選擇 Perplexity (困惑度)",
                    min_value=5,
                    max_value=min(50, len(cleaned_df)-1), # Perplexity must be less than n_samples
                    value=30,
                    step=1,
                    key="tsne_perplexity",
                    help="Perplexity 關係到每個點考慮的近鄰數量，建議值在 5 到 50 之間。"
                )

                learning_rate = st.slider(
                    "選擇 Learning Rate (學習率)",
                    min_value=10,
                    max_value=1000,
                    value=200,
                    step=10,
                    key="tsne_learning_rate",
                    help="學習率控制每次迭代時點位置移動的步伐大小，預設值為 200。"
                )

                if st.button("執行 t-SNE"):
                    with st.spinner("執行 t-SNE 及繪製圖表中... (這可能需要一段時間)"):
                        try:
                            # Execute t-SNE
                            tsne_df = preprocessor.apply_tsne(numerical_cols, perplexity, learning_rate)

                            # Display scatter plot
                            st.markdown("##### t-SNE 2D 散佈圖")
                            tsne_scatter_base64 = plot_tsne_2d_scatter(tsne_df, cleaned_df, 'Electricity_Usage')
                            st.image(f"data:image/png;base64,{tsne_scatter_base64}", use_container_width=True)
                            
                            with st.expander("📊 結論：t-SNE 分析"):
                                st.markdown(generate_tsne_conclusion(perplexity, learning_rate), unsafe_allow_html=True)

                            with st.expander("🔎 查看 t-SNE 轉換後的數據"):
                                st.dataframe(tsne_df.style.format('{:.4f}'))

                        except Exception as e:
                            st.error(f"執行 t-SNE 時發生錯誤: {e}")

    with sub_tab1_2:
        st.subheader("特徵排序與距離度量 (Feature Ranking & Distance Metrics)")
        st.markdown(textwrap.dedent("""
            <div class="info-box">
            <ul>
                <li>特徵排序旨在評估各特徵對目標變數的重要性，有助於識別出對模型預測最具影響力的特徵</li>
                <li>距離度量用於量化資料點之間的相似性或差異性，是許多聚類、分類和異常檢測算法的基礎</li>
            </ul>
            </div>
            """), unsafe_allow_html=True)
        
        # Initialize session state for this sub-tab
        if 'fr_num_result' not in st.session_state:
            st.session_state.fr_num_result = None
        if 'fr_cat_result' not in st.session_state:
            st.session_state.fr_cat_result = None
        if 'dist_num_result' not in st.session_state:
            st.session_state.dist_num_result = None
        if 'dist_ham_result' not in st.session_state:
            st.session_state.dist_ham_result = None

        if st.button("🧹 清除所有分析結果", key="clear_fr_dist_results"):
            st.session_state.fr_num_result = None
            st.session_state.fr_cat_result = None
            st.session_state.dist_num_result = None
            st.session_state.dist_ham_result = None
            st.rerun()

        st.markdown("#### 🎡 數值型特徵排序 (Numerical Feature Ranking)")
        if not numerical_cols:
            st.warning("資料中沒有足夠的數值型特徵進行排序分析")
        else:
            ranking_method = st.selectbox(
                "選擇排序方法：",
                options=["互資訊 (Mutual Information)", "資訊增益 (Information Gain)"],
                key="numerical_ranking_method"
            )

            ig_bins_slider = None
            if ranking_method == "資訊增益 (Information Gain)":
                ig_bins_slider = st.slider(
                    "選擇目標變數（用電量）的離散化箱數",
                    min_value=2,
                    max_value=10,
                    value=5,
                    step=1,
                    key="ig_bins"
                )

            if st.button("執行特徵排序", key="run_numerical_ranking"):
                with st.spinner(f"執行 {ranking_method} 計算中..."):
                    try:
                        feature_importances = None
                        if ranking_method == "互資訊 (Mutual Information)":
                            feature_importances = preprocessor.calculate_mutual_info(
                                numerical_cols, 'Electricity_Usage'
                            )
                        elif ranking_method == "資訊增益 (Information Gain)":
                            feature_importances = preprocessor.calculate_information_gain(
                                numerical_cols, 'Electricity_Usage', bins=ig_bins_slider
                            )
                            
                        if feature_importances is not None and not feature_importances.empty:
                            st.session_state.fr_num_result = {
                                'method': ranking_method,
                                'chart': plot_feature_importance_bar_chart(feature_importances, f"{ranking_method} 特徵重要性", color='#3CBBDE'),
                                'conclusion': generate_feature_ranking_conclusion(feature_importances, ranking_method)
                            }
                        else:
                            st.info("沒有可顯示的特徵重要性結果。" )
                            st.session_state.fr_num_result = None
                        st.rerun()

                    except Exception as e:
                        st.error(f"執行特徵排序時發生錯誤: {e}")
                        st.session_state.fr_num_result = None
            
            if st.session_state.fr_num_result:
                result = st.session_state.fr_num_result
                st.markdown(f"##### 特徵重要性 ({result['method']})")
                st.image(f"data:image/png;base64,{result['chart']}", use_container_width=True)
                with st.expander("📊 結論：數值型特徵排序"):
                    st.markdown(result['conclusion'], unsafe_allow_html=True)


        st.markdown("---")
        st.markdown("#### 🎪 類別型特徵排序 (Categorical Feature Ranking)")
        categorical_cols_for_ranking = [col for col in ['Science_Park', 'Sub_Science_Park', 'County', 'Town'] if col in cleaned_df.columns]
        if not categorical_cols_for_ranking:
            st.warning("資料中沒有類別型特徵可進行排序分析。" )
        else:
            cat_ig_bins = st.slider(
                "選擇目標變數（用電量）的離散化箱數",
                min_value=2,
                max_value=10,
                value=5,
                step=1,
                key="cat_ig_bins"
            )
            if st.button("執行類別特徵排序", key="run_categorical_ranking"):
                with st.spinner("執行資訊增益計算中..."):
                    try:
                        cat_feature_importances = preprocessor.calculate_information_gain(
                            categorical_cols_for_ranking, 'Electricity_Usage', bins=cat_ig_bins
                        )
                        if not cat_feature_importances.empty:
                            st.session_state.fr_cat_result = {
                                'chart': plot_feature_importance_bar_chart(cat_feature_importances, "類別型特徵重要性 (資訊增益)", color='#9FCE63'),
                                'conclusion': generate_feature_ranking_conclusion(cat_feature_importances, "資訊增益 (Information Gain)")
                            }
                        else:
                            st.info("沒有可顯示的特徵重要性結果。" )
                            st.session_state.fr_cat_result = None
                        st.rerun()
                    except Exception as e:
                        st.error(f"執行類別特徵排序時發生錯誤: {e}")
                        st.session_state.fr_cat_result = None
            
            if st.session_state.fr_cat_result:
                result = st.session_state.fr_cat_result
                st.markdown("##### 特徵重要性 (資訊增益)")
                st.image(f"data:image/png;base64,{result['chart']}", use_container_width=True)
                with st.expander("📊 結論：類別型特徵排序"):
                    st.markdown(result['conclusion'], unsafe_allow_html=True)


        st.markdown("---")
        st.markdown("#### 🎢 數值距離度量 (Numerical Distance Metrics)")
        st.markdown(textwrap.dedent("""
            <div class="info-box">
            <ul>
                <li>距離度量用於量化資料點之間的相似性或差異性，是許多機器學習算法（如：KNN、聚類）的基礎</li>
                <li>不同的距離公式對特徵的尺度和資料的分佈有不同的敏感度，選擇合適的度量方式至關重要</li>
            </ul>
            </div>
            """), unsafe_allow_html=True)
        
        if not numerical_cols:
            st.warning("資料中沒有數值型特徵可供計算距離。")
        else:
            col1, col2 = st.columns(2)
            with col1:
                idx1 = st.number_input("選擇資料點 1 的索引 (Index)：", min_value=cleaned_df.index.min(), max_value=cleaned_df.index.max(), value=cleaned_df.index.min(), key="dist_idx1")
            with col2:
                idx2 = st.number_input("選擇資料點 2 的索引 (Index)：", min_value=cleaned_df.index.min(), max_value=cleaned_df.index.max(), value=cleaned_df.index.max(), key="dist_idx2")

            features_for_dist = st.multiselect(
                "選擇要納入計算的數值特徵：",
                options=numerical_cols,
                default=numerical_cols,
                key="dist_features"
            )

            p_minkowski = st.slider("設定敏可夫斯基 (Minkowski) 距離的 p 值：", min_value=1, max_value=10, value=3, key="dist_p_minkowski")

            if st.button("計算距離", key="run_dist_calc"):
                if idx1 == idx2:
                    st.error("請選擇兩個不同的資料點進行比較。" )
                elif not features_for_dist:
                    st.error("請至少選擇一個特徵進行計算。" )
                else:
                    try:
                        with st.spinner("計算距離中..."):
                            selected_points_df = cleaned_df.loc[[idx1, idx2], features_for_dist]
                            distances = preprocessor.calculate_distance_metrics(idx1, idx2, features_for_dist, p_minkowski)
                            st.session_state.dist_num_result = {
                                'points': selected_points_df,
                                'distances': distances,
                                'p_minkowski': p_minkowski
                            }
                        st.rerun()
                    except Exception as e:
                        st.error(f"計算距離時發生錯誤: {e}")
                        st.session_state.dist_num_result = None

            if st.session_state.dist_num_result:
                result = st.session_state.dist_num_result
                st.markdown("##### 🪢 已選資料點之特徵比較")
                st.dataframe(result['points'])
                st.markdown("##### ✨ 距離計算結果")
                res_col1, res_col2, res_col3, res_col4 = st.columns(4)
                with res_col1:
                    st.metric(label="歐幾里得 (Euclidean)", value=f"{result['distances']['Euclidean']:.4f}")
                with res_col2:
                    st.metric(label="曼哈頓 (Manhattan)", value=f"{result['distances']['Manhattan']:.4f}")
                with res_col3:
                    st.metric(label="切比雪夫 (Chebyshev)", value=f"{result['distances']['Chebyshev']:.4f}")
                with res_col4:
                    st.metric(label=f"敏可夫斯基 (Minkowski, p={result['p_minkowski']})", value=f"{result['distances']['Minkowski']:.4f}")

                with st.expander("📌 距離度量方法定義與特性說明"):
                    st.markdown(textwrap.dedent(f"""
                        - **歐幾里得距離 (Euclidean Distance)**：
                          - **定義**：兩點在多維空間中的「直線距離」，即最直觀理解的距離
                          - **公式**：`√Σ(x_i - y_i)²`
                          - **特性**：若某個特徵的尺度遠大於其他特徵，它將主導距離的計算結果，故在使用歐幾里得距離前，通常建議進行**特徵標準化**

                        - **曼哈頓距離 (Manhattan Distance / City Block)**：
                          - **定義**：想像在棋盤格狀的城市中，從 A 點到 B 點只能沿著格線走，不能斜穿，累計需要走過的總路徑長
                          - **公式**：`Σ|x_i - y_i|`
                          - **特性**：計算各維度座標差的絕對值總和，相較於歐幾里得距離，它對異常值（outliers）較不敏感

                        - **切比雪夫距離 (Chebyshev Distance)**：
                          - **定義**：各維度座標差的「最大值」
                          - **公式**：`max|x_i - y_i|`
                          - **特性**：只考慮差異最大的那個維度，其他維度的差異都被忽略，適用於衡量「最壞情況」下的差異

                        - **敏可夫斯基距離 (Minkowski Distance)**：
                          - **定義**：一個通用的距離公式，歐幾里得距離和曼哈頓距離都是它的特例
                          - **公式**：`(Σ|x_i - y_i|^p)^(1/p)`
                          - **特性**：
                            - 當 `p=1` 時，等同於**曼哈頓距離**
                            - 當 `p=2` 時，等同於**歐幾里得距離**
                            - 您選擇的 `p={result['p_minkowski']}`。隨著 `p` 值增大，此距離會愈來愈接近**切比雪夫距離**
                    """), unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 🎠 漢明距離 (Hamming Distance)")
        st.markdown(textwrap.dedent("""
            <div class="info-box">
            <ul>
                <li>漢明距離用於計算兩個等長字串之間，對應位置上不同字元的數量</li>
                <li>在類別特徵的脈絡下，它衡量兩個資料點在各個類別特徵上的「不相似程度」</li>
                <li>距離為 0 表示兩個資料點在所有被選的類別特徵上都完全相同</li>
            </ul>
            </div>
            """), unsafe_allow_html=True)

        categorical_cols = ['Science_Park', 'Sub_Science_Park', 'County', 'Town']
        if not any(col in cleaned_df.columns for col in categorical_cols):
            st.warning("資料中沒有類別型特徵可供計算漢明距離。" )
        else:
            col1_ham, col2_ham = st.columns(2)
            with col1_ham:
                idx1_ham = st.number_input("選擇資料點 1 的索引 (Index)：", min_value=cleaned_df.index.min(), max_value=cleaned_df.index.max(), value=cleaned_df.index.min(), key="ham_idx1")
            with col2_ham:
                idx2_ham = st.number_input("選擇資料點 2 的索引 (Index)：", min_value=cleaned_df.index.min(), max_value=cleaned_df.index.max(), value=cleaned_df.index.max(), key="ham_idx2")

            features_for_ham = st.multiselect(
                "選擇要納入計算的類別特徵：",
                options=categorical_cols,
                default=[col for col in categorical_cols if col in cleaned_df.columns],
                key="ham_features"
            )

            if st.button("計算漢明距離", key="run_ham_calc"):
                if idx1_ham == idx2_ham:
                    st.error("請選擇兩個不同的資料點進行比較。" )
                elif not features_for_ham:
                    st.error("請至少選擇一個特徵進行計算。" )
                else:
                    try:
                        with st.spinner("計算漢明距離中..."):
                            p1_decoded = get_decoded_categorical_values(cleaned_df.loc[idx1_ham], features_for_ham, preprocessor.encoders)
                            p2_decoded = get_decoded_categorical_values(cleaned_df.loc[idx2_ham], features_for_ham, preprocessor.encoders)
                            
                            comparison_df = pd.DataFrame({
                                f'Index {idx1_ham}': p1_decoded,
                                f'Index {idx2_ham}': p2_decoded
                            })

                            ham_dist_result = preprocessor.calculate_hamming_distance(idx1_ham, idx2_ham, features_for_ham)
                            
                            st.session_state.dist_ham_result = {
                                'comparison_df': comparison_df,
                                'ham_dist_result': ham_dist_result,
                                'idx1': idx1_ham,
                                'idx2': idx2_ham
                            }
                        st.rerun()
                    except Exception as e:
                        st.error(f"計算漢明距離時發生錯誤: {e}")
                        st.session_state.dist_ham_result = None

            if st.session_state.dist_ham_result:
                result = st.session_state.dist_ham_result
                st.markdown("##### 🪢 已選資料點之特徵比較 (原始類別)")
                st.dataframe(result['comparison_df'])

                st.markdown("##### ✨ 漢明距離計算結果")
                st.metric(
                    label="漢明距離 (Hamming Distance)", 
                    value=result['ham_dist_result']['Hamming Distance'],
                    help=f"在 {result['ham_dist_result']['Compared Features']} 個被比較的特徵中，有 {result['ham_dist_result']['Hamming Distance']} 個特徵的值不相同"
                )

                with st.expander("📌 漢明距離方法定義與特性說明"):
                    st.markdown(textwrap.dedent(f"""
                        - **計算方式**：
                          - 比較「Index <span style='color:#4481D7'>**{result['idx1']}**</span>」和「Index <span style='color:#4481D7'>**{result['idx2']}**</span>」在所選取的 <span style='color:#4481D7'>**{result['ham_dist_result']['Compared Features']}**</span> 個類別特徵上的值
                          - 計算出其中有 <span style='color:#4481D7'>**{result['ham_dist_result']['Hamming Distance']}**</span> 個特徵值不相同
                        - **意義**：
                          - 漢明距離衡量兩個樣本在特徵（類別屬性）上的差異程度
                          - 距離愈大，表示這兩個樣本的「輪廓」或「屬性」愈不相似
                          - 例如：兩個樣本的特徵值分別為「(新竹園區、新竹市)」和「(台中園區、台中市)」，兩筆資料的特徵值均不同，則漢明距離即為 2
                    """), unsafe_allow_html=True)


with tab2:
    st.header("數量縮減 (Numerosity Reduction)")
    st.markdown(textwrap.dedent("""
        <div class="info-box">
        <ul>
            <li>數量縮減旨在以較小的資料表示形式（如模型參數、統計摘要或抽樣）替代原始大量資料，同時盡量保留其本質特徵</li>
            <li>這有助於提高資料處理效率、降低儲存成本，並加速模型訓練</li>
        </ul>
        </div>
        """), unsafe_allow_html=True)

    reduction_type = st.radio(
        "選擇縮減方法類型：",
        options=["參數方法 (Parametric Methods)", "非參數方法 (Non-parametric Methods)"],
        key="reduction_type"
    )

    if reduction_type == "參數方法 (Parametric Methods)":
        numerosity_reduction_method = st.radio(
            "選擇參數化精簡方法：",
            options=["線性迴歸 (Linear Regression)", "決策樹迴歸 (Decision Tree Regression)"],
            key="numerosity_reduction_method"
        )

        numerical_cols = cleaned_df.select_dtypes(include=np.number).columns.tolist()
        if not numerical_cols:
            st.warning("資料中沒有數值型特徵可供進行迴歸分析。" )
        else:
            col_x, col_y = st.columns(2)
            with col_x:
                feature_col = st.selectbox(
                    "選擇自變數 (X 軸特徵):",
                    options=numerical_cols,
                    index=numerical_cols.index('Avg_Temperature') if 'Avg_Temperature' in numerical_cols else 0,
                    key="param_red_feature_col"
                )
            with col_y:
                target_col = st.selectbox(
                    "選擇應變數 (Y 軸特徵):",
                    options=numerical_cols,
                    index=numerical_cols.index('Electricity_Usage') if 'Electricity_Usage' in numerical_cols else 0,
                    key="param_red_target_col"
                )

            if numerosity_reduction_method == "線性迴歸 (Linear Regression)":
                st.markdown("### 線性迴歸模型參數化精簡 (Parametric Reduction via Linear Regression Model)")
                st.info("線性迴歸模型透過少數參數（如截距和係數）來捕捉變數之間的關係，從而「精簡」大量資料，可以使用這些參數來重建或預測資料，而無需保留所有原始資料點", icon="ℹ️")

                if st.button("執行線性迴歸分析並精簡", key="run_linear_regression_reduction"):
                    if feature_col == target_col:
                        st.error("自變數和應變數不能是同一個特徵。" )
                    else:
                        with st.spinner("執行線性迴歸分析中..."):
                            try:
                                regression_results = preprocessor.perform_linear_regression(feature_col, target_col)
                                model = regression_results['model']
                                coefficient = regression_results['coefficient']
                                intercept = regression_results['intercept']
                                r_squared = regression_results['r_squared']

                                # Plotting
                                fig, ax = plt.subplots(figsize=(10, 6))
                                sns.scatterplot(x=cleaned_df[feature_col], y=cleaned_df[target_col], ax=ax, alpha=0.6, color='#3CBBDE', label='資料點')
                                # Plot regression line
                                x_plot = np.array([cleaned_df[feature_col].min(), cleaned_df[feature_col].max()])
                                y_plot = intercept + coefficient * x_plot
                                ax.plot(x_plot, y_plot, color='#DD6D6A', linewidth=2, label=f'迴歸線: Y = {coefficient:.2f}X + {intercept:.2f}')
                                
                                ax.set_title(f'{feature_col} 與 {target_col} 的線性迴歸')
                                ax.set_xlabel(feature_col)
                                ax.set_ylabel(target_col)
                                ax.legend()
                                plt.tight_layout()
                                buf = io.BytesIO()
                                plt.savefig(buf, format='png')
                                plt.close(fig)
                                regression_plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                                st.image(f"data:image/png;base64,{regression_plot_base64}", use_container_width=True)

                                st.markdown("##### 迴歸模型參數")
                                col_coeff, col_intercept, col_r2 = st.columns(3)
                                with col_coeff:
                                    st.metric(label="斜率 (Slope/Coefficient)", value=f"{coefficient:.4f}")
                                with col_intercept:
                                    st.metric(label="截距 (Intercept)", value=f"{intercept:.4f}")
                                with col_r2:
                                    st.metric(label="R-squared (R²)", value=f"{r_squared:.4%}")
                                
                                with st.expander("📊 結論：線性迴歸模型參數化精簡"):
                                    st.markdown(textwrap.dedent(f"""
                                        - **精簡原理**：
                                          - 傳統上，要描述 <span style='color:#4481D7'>**{feature_col}**</span> 與 <span style='color:#4481D7'>**{target_col}**</span> 之間的關係，需要儲存所有的資料點
                                          - 透過線性迴歸，可以用一個簡單方程式 <span style='color:#4481D7'>**Y = {coefficient:.2f}X + {intercept:.2f}**</span> 來「精簡」這種關係
                                          - 意謂：無須儲存每個原始資料點，只需儲存這幾個**模型參數** (<span style='color:#4481D7'>**斜率**</span> 和 <span style='color:#4481D7'>**截距**</span>) 就可代表原始資料中蘊含的線性趨勢

                                        - **表示能力**：
                                          - <span style='color:#4481D7'>**R-squared (R²)**</span> = <span style='color:#4481D7'>**{r_squared:.4%}**</span>，表示此模型解釋應變數 <span style='color:#4481D7'>**{target_col}**</span> 約 <span style='color:#4481D7'>**{r_squared:.4%}**</span> 的變異
                                          - 較高的 R-squared 值，意指模型參數能更好地代表原始資料的關係以實現更有效的數量精簡

                                        - **實用性**：
                                          - 在不損失過多資訊的情況下，可以減少資料儲存和傳輸的需求
                                          - 這種精簡後的表達形式更易於理解，與應用於未來的預測能力
                                    """), unsafe_allow_html=True)
                            
                            except Exception as e:
                                st.error(f"執行線性迴歸分析時發生錯誤: {e}")

            elif numerosity_reduction_method == "決策樹迴歸 (Decision Tree Regression)":
                st.markdown("### 決策樹迴歸模型參數化精簡 (Parametric Reduction via Decision Tree Regression Model)")
                st.info("決策樹迴歸模型透過樹狀結構來捕捉資料中的非線性關係，其樹的節點、分支和葉子節點的規則（即模型結構）本身就是資料的精簡表示", icon="ℹ️")
                
                max_depth = st.slider(
                    "選擇決策樹的最大深度 (max_depth)：",
                    min_value=1,
                    max_value=10,
                    value=5,
                    step=1,
                    key="dt_max_depth",
                    help="控制決策樹的複雜度。深度愈大，模型愈複雜，但過深可能導致過度擬合"
                )

                if st.button("執行決策樹迴歸分析並精簡", key="run_dt_regression_reduction"):
                    if feature_col == target_col:
                        st.error("自變數和應變數不能是同一個特徵。" )
                    else:
                        with st.spinner("執行決策樹迴歸分析中..."):
                            try:
                                dt_results = preprocessor.perform_decision_tree_regression(feature_col, target_col, max_depth=max_depth)
                                dt_model = dt_results['model']
                                dt_r_squared = dt_results['r_squared']

                                # Plotting
                                fig, ax = plt.subplots(figsize=(10, 6))
                                sns.scatterplot(x=cleaned_df[feature_col], y=cleaned_df[target_col], ax=ax, alpha=0.6, color='#9FCE63', label='資料點')
                                
                                # For decision tree, predict over a range to show the step-like function
                                x_range = np.linspace(cleaned_df[feature_col].min(), cleaned_df[feature_col].max(), 500).reshape(-1, 1)
                                y_pred_dt = dt_model.predict(x_range)
                                ax.plot(x_range, y_pred_dt, color='#DD6D6A', linewidth=2, label=f'決策樹迴歸 (Max Depth: {max_depth})')
                                
                                ax.set_title(f'{feature_col} 與 {target_col} 的決策樹迴歸')
                                ax.set_xlabel(feature_col)
                                ax.set_ylabel(target_col)
                                ax.legend()
                                plt.tight_layout()
                                buf = io.BytesIO()
                                plt.savefig(buf, format='png')
                                plt.close(fig)
                                dt_regression_plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                                st.image(f"data:image/png;base64,{dt_regression_plot_base64}", use_container_width=True)

                                st.markdown("##### 迴歸模型評估")
                                st.metric(label="R-squared (R²)", value=f"{dt_r_squared:.4%}")
                                
                                with st.expander("📊 結論：決策樹迴歸模型參數化精簡"):
                                    st.markdown(textwrap.dedent(f"""
                                        - **精簡原理**：
                                          - 決策樹迴歸模型不是透過線性方程式，而是透過一系列的「分支規則」和「最終葉節點的值」來對資料進行分區和預測
                                          - 整個**樹狀結構本身**（包括每個節點的分割條件、分割特徵和每個葉節點的預測值）就是資料的一種精簡表示
                                          - 無須儲存所有原始資料點，只需儲存這棵決策樹的結構，就能夠根據 <span style='color:#4481D7'>**{feature_col}**</span> 的值來預測 <span style='color:#4481D7'>**{target_col}**</span>

                                        - **表示能力**：
                                          - <span style='color:#4481D7'>**R-squared**</span> = <span style='color:#4481D7'>{dt_r_squared:.4%}**</span>，表示此模型解釋應變數 <span style='color:#4481D7'>**{target_col}**</span> 約 <span style='color:#4481D7'>**{dt_r_squared:.4%}**</span> 的變異
                                          - 決策樹在捕捉「非線性關係」方面通常比線性迴歸更靈活
                                          - <span style='color:#4481D7'>**最大深度 (max_depth)**</span> 為 <span style='color:#4481D7'>**{max_depth}**</span> 控制樹的複雜度，也間接影響精簡的程度
                                             - 深度愈淺，精簡程度愈高，但可能損失更多細節
                                             - 深度愈深，模型愈能捕捉複雜關係，但精簡程度相對較低且可能過度擬合

                                        - **實用性**：
                                          - 決策樹在處理具有複雜的「非線性模型」資料時，能提供比線性模型更好的資料摘要能力
                                          - 透過調整 `max_depth`，可以在精簡程度和模型解釋力之間取得平衡
                                    """), unsafe_allow_html=True)
                            
                            except Exception as e:
                                st.error(f"執行決策樹迴歸分析時發生錯誤: {e}")

    elif reduction_type == "非參數方法 (Non-parametric Methods)":
        non_parametric_method = st.radio(
            "選擇非參數化精簡方法：",
            options=["直方圖 (Histogram)", "叢集 (Clustering)", "抽樣 (Sampling)"],
            key="non_parametric_method_select"
        )

        numerical_cols = cleaned_df.select_dtypes(include=np.number).columns.tolist()
        if not numerical_cols:
            st.warning("資料中沒有數值型特徵可供分析。", icon="⚠️")
        else:
            if non_parametric_method == "直方圖 (Histogram)":
                st.markdown("### 直方圖數量精簡 (Numerosity Reduction via Histogram)")
                st.info("直方圖是一種非參數方法，它將資料分組成一系列的「箱 (bins)」並計算每個箱中的資料點數量，無須儲存每個原始資料點，只需儲存每個箱的邊界和它所包含的計數，從而達到資料精簡的目的", icon="ℹ️")

                hist_feature_col = st.selectbox(
                    "選擇要建立直方圖的特徵：",
                    options=numerical_cols,
                    index=numerical_cols.index('Electricity_Usage') if 'Electricity_Usage' in numerical_cols else 0,
                    key="hist_feature_col_histogram"
                )

                hist_bins = st.slider(
                    "選擇直方圖的箱數 (Number of Bins)：",
                    min_value=2,
                    max_value=50,
                    value=10,
                    step=1,
                    key="hist_bins_histogram",
                    help="箱數愈多，資料的表示愈精確，但精簡效果愈差；箱數愈少，精簡效果愈好，但可能損失更多細節。"
                )

                if st.button("執行直方圖分析並精簡", key="run_histogram_reduction_histogram"):
                    with st.spinner("建立直方圖中..."):
                        try:
                            # Create histogram
                            histogram_df = preprocessor.create_histogram_bins(hist_feature_col, hist_bins)
                            
                            st.markdown("##### 直方圖")
                            
                            # Rename for st.bar_chart and st.dataframe
                            chart_df = histogram_df.reset_index()
                            chart_df.columns = [hist_feature_col, 'Count']
                            
                            st.bar_chart(chart_df.set_index(hist_feature_col), color="#F5C65D", x_label=hist_feature_col, y_label='Count', height=400)
                                                        
                            with st.expander("📊 結論：直方圖數量精簡"):
                                st.markdown(textwrap.dedent(f"""
                                    - **精簡原理**：
                                      - 原始的 <span style='color:#4481D7'>**{hist_feature_col}**</span> 特徵包含 <span style='color:#4481D7'>**{len(cleaned_df)}**</span> 個資料點
                                      - 透過直方圖，將這些資料點分佈到 <span style='color:#4481D7'>**{hist_bins}**</span> 個箱子中
                                      - 無須儲存全部的原始資料，只需儲存這 <span style='color:#4481D7'>**{hist_bins}**</span> 個箱子的「邊界」和每個箱子的「計數」即可
                                    - **表示能力**：
                                      - 直方圖提供一個關於資料分佈的緊湊摘要
                                      - 快速了解資料的統計分佈之集中趨勢、離散程度，以及是否存在偏態
                                    - **權衡 (Trade-off)**：
                                      - 此法的代價是損失資料的「個體細節」，只能知道有多少個值落入某個區間，但無法得知它們的確切數值
                                      - 「箱數的選擇」至關重要：
                                         - 箱數太 <span style='color:#4481D7'>少</span> 會過度簡化而損失過多資訊
                                         - 箱數太 <span style='color:#4481D7'>多</span> 則會降低精簡效果，且可能導致過度擬合資料中的噪音
                                """), unsafe_allow_html=True)
                            
                            with st.expander("🔎 查看直方圖數據"):
                                st.dataframe(chart_df)

                            with st.expander("⛳ 方法比較：直方圖、等寬裝箱與等深裝箱"):
                                st.markdown("""
                                    <div class="info-box">
                                    <p>此處的「直方圖」與「資料轉換」頁面中的「等寬裝箱法」皆使用相同的技術，但兩者的應用目標與呈現方式不同，以下為三者的比較：</p>
                                    <ul>
                                        <li>
                                            <strong>直方圖 (於此頁面)</strong>
                                            <ul>
                                                <li><strong>目標</strong>：<span style='color:#4481D7'>數量縮減 (Numerosity Reduction)</span></li>
                                                <li><strong>原理</strong>：將大量的原始資料點，摘要成少數幾個「箱子區間」與對應的「資料點計數」，只需儲存這些摘要資訊即可減少資料量</li>
                                                <li><strong>實現方式</strong>：此處的直方圖是基於「等寬裝箱法」來實現</li>
                                            </ul>
                                        </li>
                                        <li>
                                            <strong>等寬裝箱法 (Equal-width Binning)</strong>
                                            <ul>
                                                <li><strong>目標</strong>：確保每個「箱子」的<strong>寬度(範圍)</strong>都相同</li>
                                                <li><strong>結果</strong>：每個箱子裡的<strong>資料點數量可能差異很大</strong>
                                                <ul>
                                                    <li>在資料密集的區間，箱內點數多</li>
                                                    <li>在資料稀疏的區間，箱內點數少</li>
                                                </ul>
                                            </ul>
                                        </li>
                                        <li>
                                            <strong>等深裝箱法 (Equal-depth Binning)</strong>
                                            <ul>
                                                <li><strong>目標</strong>：控制每個「箱子」裡的<strong>資料點數量</strong>大致相同</li>
                                                <li><strong>結果</strong>：每個箱子的<strong>寬度(範圍)通常會不同</strong>
                                                <ul>
                                                    <li>在資料密集的區間，箱子的寬度會變得很窄</li>
                                                    <li>在資料稀疏的區間，箱子的寬度會變得很寬</li>
                                                </ul>
                                            </ul>
                                        </li>
                                    </ul>
                                    </div>
                                """, unsafe_allow_html=True)

                        except Exception as e:
                            st.error(f"建立直方圖時發生錯誤：{e}", icon="🚫")
            
            elif non_parametric_method == "叢集 (Clustering)":
                st.markdown("### 叢集數量精簡 (Numerosity Reduction via Clustering)")
                st.info("叢集技術（如 K-Means）透過將相似的 N 個資料點分組到 K 個群體中，並用每個群體的「質心 (Centroid)」來代表該群體中的所有資料點", icon="ℹ️")

                features_for_clustering = st.multiselect(
                    "選擇要進行叢集分析的特徵 (至少選擇 2 個數值型特徵)：",
                    options=numerical_cols,
                    default=[c for c in ['Avg_Temperature', 'Electricity_Usage'] if c in numerical_cols][:2],
                    key="clustering_features_clustering"
                )

                n_clusters = st.slider(
                    "選擇叢集數量 (K)：",
                    min_value=2,
                    max_value=min(10, len(cleaned_df) - 1),
                    value=3,
                    step=1,
                    key="n_clusters_clustering",
                    help="K 值決定資料點被精簡成的數量。K 愈小，精簡程度愈高，但可能損失更多細節；K 愈大，精簡程度愈低，但能更好地保留資料結構。"
                )

                if st.button("執行叢集分析並精簡", key="run_clustering_reduction_clustering"):

                    if len(features_for_clustering) < 2:
                        st.error("請至少選擇 2 個數值型特徵進行叢集分析。", icon="🚫")
                    else:
                        with st.spinner("執行叢集分析中..."):
                            try:
                                clustering_results = preprocessor.perform_clustering_reduction(features_for_clustering, n_clusters)
                                cluster_labels = clustering_results['cluster_labels']
                                cluster_centroids = clustering_results['cluster_centroids']
                                descriptive_labels = clustering_results['descriptive_labels']
                                
                                # Add cluster labels to the cleaned_df for plotting
                                plot_df = cleaned_df.copy()
                                plot_df['Cluster_Label'] = [descriptive_labels[label] for label in cluster_labels]
                                
                                # Ensure Cluster_Label is an ordered categorical type for correct legend order
                                ordered_cluster_labels = [descriptive_labels[i] for i in sorted(descriptive_labels.keys())]
                                plot_df['Cluster_Label'] = pd.Categorical(
                                    plot_df['Cluster_Label'],
                                    categories=ordered_cluster_labels,
                                    ordered=True
                                )
                                
                                # Plotting: Scatter plot of original data points and centroids
                                fig, ax = plt.subplots(figsize=(10, 8))
                                if len(features_for_clustering) >= 2:
                                    sns.scatterplot(
                                        x=features_for_clustering[0],
                                        y=features_for_clustering[1],
                                        hue='Cluster_Label',
                                        data=plot_df,
                                        palette=sns.color_palette("bright"),
                                        alpha=0.6,
                                        ax=ax,
                                        legend='full'
                                    )
                                    # Plot centroids
                                    ax.scatter(
                                        cluster_centroids[features_for_clustering[0]],
                                        cluster_centroids[features_for_clustering[1]],
                                        marker='X',
                                        s=200,
                                        color='red',
                                        label='Centroids',
                                        edgecolor='black'
                                    )
                                    ax.set_title(f'叢集分析結果 ({features_for_clustering[0]} vs {features_for_clustering[1]})')
                                    ax.set_xlabel(features_for_clustering[0])
                                    ax.set_ylabel(features_for_clustering[1])
                                    ax.legend()
                                elif len(features_for_clustering) == 1:
                                    sns.histplot(x=features_for_clustering[0], hue='Cluster_Label', data=plot_df, kde=True, palette='viridis', ax=ax)
                                    ax.scatter(
                                        cluster_centroids[features_for_clustering[0]],
                                        [0] * len(cluster_centroids), # Centroids at y=0 for 1D plot
                                        marker='X',
                                        s=200,
                                        color='red',
                                        label='Centroids',
                                        edgecolor='black'
                                    )
                                    ax.set_title(f'叢集分析結果 ({features_for_clustering[0]})')
                                    ax.set_xlabel(features_for_clustering[0])
                                    ax.set_ylabel('密度')
                                    ax.legend()
                                else:
                                    st.warning("請選擇至少一個特徵進行叢集分析。", icon="⚠️")
                                
                                plt.tight_layout()
                                buf = io.BytesIO()
                                plt.savefig(buf, format='png')
                                plt.close(fig)
                                clustering_plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                                st.image(f"data:image/png;base64,{clustering_plot_base64}", use_container_width=True)
                                
                                st.markdown("##### 叢集質心 (Centroids)")
                                formatter = {col: '{:.4f}' for col in features_for_clustering}
                                st.dataframe(cluster_centroids.style.format(formatter))
                                
                                # Generate the descriptive labels string with correct indentation
                                label_list_md = "\n".join([f"                                          - <span style='color:#4481D7'>**{label}**</span>" for label in descriptive_labels.values()])

                                with st.expander("📊 結論：叢集數量精簡"):
                                    st.markdown(textwrap.dedent(f"""
                                        - **精簡原理**：
                                          - 原始資料中包含 <span style='color:#4481D7'>**{len(cleaned_df)}**</span> 個資料點
                                          - 透過叢集分析，將這些資料點歸納為 <span style='color:#4481D7'>**{n_clusters}**</span> 個群體，並用每個群體的**質心 (Centroid)** 來代表該群體中的所有資料點
                                          - 無須儲存全部的原始資料，只需儲存這 <span style='color:#4481D7'>**{n_clusters}**</span> 個質心的座標，即可大幅減少資料量
                                        - **表示能力**：
                                          - 叢集質心能夠捕獲原始資料集中分佈的「主要模式」或「中心趨勢」
                                          - 從上圖散佈圖可以看到資料點被分組，並且每個叢集有一個紅色的 <span style='color:#DD6D6A'>**X**</span> 標記代表其質心
                                        - **叢集標籤說明**：
{label_list_md}
                                        - **權衡 (Trade-off)**：
                                          - 此法的代價是損失資料的「細微差異」，每個資料點都被視為與其所屬質心相同，忽略叢集內部的變異
                                          - 「K 值的選擇」至關重要：
                                             - K 值太 <span style='color:#4481D7'>小</span> 會導致過度概括，損失重要資訊
                                             - K 值太 <span style='color:#4481D7'>大</span> 會降低精簡效果，且可能導致過度擬合資料中的噪音
                                    """), unsafe_allow_html=True)
                                
                                # Generate the descriptive labels string with correct indentation
                                label_list_string = "".join([f"- <span style='color:#4481D7'>**{label}**</span>\n" for label in descriptive_labels.values()])
                                
                                
                            except Exception as e:
                                st.error(f"執行叢集分析時發生錯誤: {e}", icon="🚫")
            
            elif non_parametric_method == "抽樣 (Sampling)":
                st.markdown("### 抽樣數量精簡 (Numerosity Reduction via Sampling)")
                st.info("抽樣是從整體資料中選取一部分子集（樣本）的過程，這個樣本可以被用來代表原始的完整資料集", icon="ℹ️")

                sampling_method = st.radio(
                    "選擇抽樣方法：",
                    options=["隨機抽樣 (Random Sampling)", "分層抽樣 (Stratified Sampling)", "系統抽樣 (Systematic Sampling)"],
                    key="sampling_method_select"
                )

                sample_size_percent = st.slider(
                    "選擇樣本大小（百分比）：", 
                    min_value=1, 
                    max_value=100, 
                    value=20, 
                    step=1,
                    key="sampling_size_percent"
                )
                sample_frac = sample_size_percent / 100.0

                stratify_col = None
                if sampling_method == "分層抽樣 (Stratified Sampling)":
                    categorical_cols = cleaned_df.select_dtypes(include=['category', 'object']).columns.tolist()
                    stratify_col = st.selectbox(
                        "選擇分層依據的類別特徵：",
                        options=categorical_cols,
                        index=categorical_cols.index('Science_Park') if 'Science_Park' in categorical_cols else 0,
                        key="stratify_by_col"
                    )

                if st.button("執行抽樣分析並精簡", key="run_sampling"):
                    try:
                        sampled_df = None
                        with st.spinner(f"執行 {sampling_method} 中..."):
                            if sampling_method == "隨機抽樣 (Random Sampling)":
                                sampled_df = preprocessor.perform_random_sampling(sample_frac)
                            elif sampling_method == "分層抽樣 (Stratified Sampling)":
                                sampled_df = preprocessor.perform_stratified_sampling(sample_frac, stratify_col)
                            elif sampling_method == "系統抽樣 (Systematic Sampling)":
                                n_step = int(1 / sample_frac)
                                sampled_df = preprocessor.perform_systematic_sampling(n_step)
                        
                        st.success(f"{sampling_method} 完成！")

                        st.markdown("##### 資料量比較")
                        col1, col2 = st.columns(2)
                        col1.metric("原始資料筆數", f"{len(cleaned_df):,} 行")
                        col2.metric("抽樣後資料筆數", f"{len(sampled_df):,} 行", delta=f"{len(sampled_df) - len(cleaned_df):,} 行")

                        st.markdown("##### 抽樣代表性評估")
                        comparison_feature = st.selectbox(
                            "選擇要比較分佈的數值特徵：",
                            options=numerical_cols,
                            index=numerical_cols.index('Electricity_Usage') if 'Electricity_Usage' in numerical_cols else 0,
                            key="sampling_comparison_feature"
                        )

                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.kdeplot(cleaned_df[comparison_feature], ax=ax, label='原始資料', color='#3CBBDE', fill=True)
                        sns.kdeplot(sampled_df[comparison_feature], ax=ax, label='抽樣資料', color='#DD6D6A', fill=True)
                        ax.set_title(f'「{comparison_feature}」- 原始資料與抽樣資料之分佈比較')
                        ax.set_xlabel(comparison_feature)
                        ax.set_ylabel('密度 (Density)')
                        ax.legend()
                        st.pyplot(fig)

                        with st.expander("📊 結論：抽樣數量精簡"):
                            if sampling_method == "隨機抽樣 (Random Sampling)":
                                st.markdown("""
                                    - **精簡原理**：
                                        - 從整體資料中完全隨機地選取樣本，每個資料點被選中的機率都相同
                                    - **優點**：
                                        - 實現最簡單、最快速，且無偏見的代表性資料
                                    - **缺點**：
                                        - 如果資料中有罕見的子群體，隨機抽樣可能無法選中足夠的樣本來代表這些子群體，導致樣本代表性不足
                                    - **適用場景**：
                                        - 當資料分佈相對均勻，或者對子群體的分析不作特別要求
                                """)
                            elif sampling_method == "分層抽樣 (Stratified Sampling)":
                                st.markdown(f"""
                                    - **精簡原理**：
                                        - 為確保每一層的樣本都具有適當的代表性，步驟有二：
                                            1. 先將資料依據某個類別特徵（此處為 <span style='color:#4481D7'>**{stratify_col}**</span>）分成數個「層」
                                            2. 接著，在每一層內部再分別進行隨機抽樣
                                    - **優點**：
                                        - 確保樣本中每個「層」的比例與原始資料中每個「層」的比例相同，從而保證樣本能更準確地反映整體的結構，特別是在少量類別數量的情況下
                                    - **缺點**：
                                        - 需要一個明確的分類特徵來進行分層，且實施起來比隨機抽樣稍微複雜
                                    - **適用場景**：
                                        1. 當資料包含重要但佔比小的子群體時
                                        2. 當需要在不同子群體間進行比較分析時
                                """, unsafe_allow_html=True)
                            elif sampling_method == "系統抽樣 (Systematic Sampling)":
                                st.markdown("""
                                    - **精簡原理**：
                                        - 也稱為「等距抽樣」
                                        - 步驟有三：
                                            1. 先計算一個抽樣間隔 `k`（例如每 10 個選 1 個）
                                            2. 再自前 `k` 個資料點中隨機選取一個作為起點
                                            3. 後續每隔 `k` 個單位選取一個樣本 
                                    - **優點**：
                                        - 操作簡單且樣本在資料中分佈均勻，確保整個資料範圍都被覆蓋到
                                    - **缺點**：
                                        - 若資料本身存在週期性，且抽樣間隔 `k` 恰好與資料的週期相同或成倍數關係，可能會導致樣本產生嚴重偏差
                                    - **適用場景**：
                                        - 當資料無明顯的週期性規律，且希望能快速、均勻地選取樣本時
                                """)

                    except Exception as e:
                        st.error(f"執行抽樣時發生錯誤: {e}")


with tab3:
    st.header("資料壓縮 (Data Compression)")
    compression_method = st.radio(
        "選擇資料壓縮方法：",
        options=["DWT (離散小波轉換)", "PCA (主成分分析)"],
        key="compression_method"
    )

    numerical_cols = cleaned_df.select_dtypes(include=np.number).columns.tolist()

    if compression_method == "DWT (離散小波轉換)":
        st.markdown("### DWT (離散小波轉換) 資料壓縮")
        st.info("DWT 將訊號分解為不同頻率的組成部分（近似係數和細節係數），透過僅保留最重要的近似係數（低頻部分）並捨棄細節係數（高頻部分），可以實現有損壓縮", icon="ℹ️")
        if not numerical_cols:
            st.warning("資料中沒有數值型特徵可供壓縮。", icon="⚠️")
        else:
            dwt_feature = st.selectbox(
                "選擇要壓縮的數值特徵：",
                options=numerical_cols,
                index=numerical_cols.index('Electricity_Usage') if 'Electricity_Usage' in numerical_cols else 0,
                key="dwt_feature"
            )

            wavelet_families = ['db', 'sym', 'coif', 'bior', 'rbio']
            wavelet_options = []
            for family in wavelet_families:
                wavelet_options.extend(pywt.wavelist(family))
                
            wavelet_name_map = {
                'haar': '哈爾小波 (Haar)',
                'db1': '多貝西小波 (Daubechies) 1', 'db2': '多貝西小波 (Daubechies) 2', 'db3': '多貝西小波 (Daubechies) 3', 'db4': '多貝西小波 (Daubechies) 4',
                'db5': '多貝西小波 (Daubechies) 5', 'db6': '多貝西小波 (Daubechies) 6', 'db7': '多貝西小波 (Daubechies) 7', 'db8': '多貝西小波 (Daubechies) 8',
                'db9': '多貝西小波 (Daubechies) 9', 'db10': '多貝西小波 (Daubechies) 10', 'db11': '多貝西小波 (Daubechies) 11', 'db12': '多貝西小波 (Daubechies) 12',
                'db13': '多貝西小波 (Daubechies) 13', 'db14': '多貝西小波 (Daubechies) 14', 'db15': '多貝西小波 (Daubechies) 15', 'db16': '多貝西小波 (Daubechies) 16',
                'db17': '多貝西小波 (Daubechies) 17', 'db18': '多貝西小波 (Daubechies) 18', 'db19': '多貝西小波 (Daubechies) 19', 'db20': '多貝西小波 (Daubechies) 20',
                'db21': '多貝西小波 (Daubechies) 21', 'db22': '多貝西小波 (Daubechies) 22', 'db23': '多貝西小波 (Daubechies) 23', 'db24': '多貝西小波 (Daubechies) 24',
                'db25': '多貝西小波 (Daubechies) 25', 'db26': '多貝西小波 (Daubechies) 26', 'db27': '多貝西小波 (Daubechies) 27', 'db28': '多貝西小波 (Daubechies) 28',
                'db29': '多貝西小波 (Daubechies) 29', 'db30': '多貝西小波 (Daubechies) 30', 'db31': '多貝西小波 (Daubechies) 31', 'db32': '多貝西小波 (Daubechies) 32',
                'db33': '多貝西小波 (Daubechies) 33', 'db34': '多貝西小波 (Daubechies) 34', 'db35': '多貝西小波 (Daubechies) 35', 'db36': '多貝西小波 (Daubechies) 36',
                'db37': '多貝西小波 (Daubechies) 37', 'db38': '多貝西小波 (Daubechies) 38',
                'sym2': '對稱小波 (Symlets) 2', 'sym3': '對稱小波 (Symlets) 3', 'sym4': '對稱小波 (Symlets) 4', 'sym5': '對稱小波 (Symlets) 5',
                'sym6': '對稱小波 (Symlets) 6', 'sym7': '對稱小波 (Symlets) 7', 'sym8': '對稱小波 (Symlets) 8', 'sym9': '對稱小波 (Symlets) 9',
                'sym10': '對稱小波 (Symlets) 10', 'sym11': '對稱小波 (Symlets) 11', 'sym12': '對稱小波 (Symlets) 12', 'sym13': '對稱小波 (Symlets) 13',
                'sym14': '對稱小波 (Symlets) 14', 'sym15': '對稱小波 (Symlets) 15', 'sym16': '對稱小波 (Symlets) 16', 'sym17': '對稱小波 (Symlets) 17',
                'sym18': '對稱小波 (Symlets) 18', 'sym19': '對稱小波 (Symlets) 19', 'sym20': '對稱小波 (Symlets) 20',
                'coif1': '科夫利特小波 (Coiflets) 1', 'coif2': '科夫利特小波 (Coiflets) 2', 'coif3': '科夫利特小波 (Coiflets) 3', 'coif4': '科夫利特小波 (Coiflets) 4',
                'coif5': '科夫利特小波 (Coiflets) 5', 'coif6': '科夫利特小波 (Coiflets) 6', 'coif7': '科夫利特小波 (Coiflets) 7', 'coif8': '科夫利特小波 (Coiflets) 8',
                'coif9': '科夫利特小波 (Coiflets) 9', 'coif10': '科夫利特小波 (Coiflets) 10', 'coif11': '科夫利特小波 (Coiflets) 11', 'coif12': '科夫利特小波 (Coiflets) 12',
                'coif13': '科夫利特小波 (Coiflets) 13', 'coif14': '科夫利特小波 (Coiflets) 14', 'coif15': '科夫利特小波 (Coiflets) 15', 'coif16': '科夫利特小波 (Coiflets) 16',
                'coif17': '科夫利特小波 (Coiflets) 17',
                'bior1.1': '雙正交小波 (Biorthogonal) 1.1', 'bior1.3': '雙正交小波 (Biorthogonal) 1.3', 'bior1.5': '雙正交小波 (Biorthogonal) 1.5',
                'bior2.2': '雙正交小波 (Biorthogonal) 2.2', 'bior2.4': '雙正交小波 (Biorthogonal) 2.4', 'bior2.6': '雙正交小波 (Biorthogonal) 2.6', 'bior2.8': '雙正交小波 (Biorthogonal) 2.8',
                'bior3.1': '雙正交小波 (Biorthogonal) 3.1', 'bior3.3': '雙正交小波 (Biorthogonal) 3.3', 'bior3.5': '雙正交小波 (Biorthogonal) 3.5', 'bior3.7': '雙正交小波 (Biorthogonal) 3.7',
                'bior3.9': '雙正交小波 (Biorthogonal) 3.9', 'bior4.4': '雙正交小波 (Biorthogonal) 4.4', 'bior5.5': '雙正交小波 (Biorthogonal) 5.5', 'bior6.8': '雙正交小波 (Biorthogonal) 6.8',
                'rbio1.1': '反向雙正交小波 (Reverse Biorthogonal) 1.1', 'rbio1.3': '反向雙正交小波 (Reverse Biorthogonal) 1.3', 'rbio1.5': '反向雙正交小波 (Reverse Biorthogonal) 1.5',
                'rbio2.2': '反向雙正交小波 (Reverse Biorthogonal) 2.2', 'rbio2.4': '反向雙正交小波 (Reverse Biorthogonal) 2.4', 'rbio2.6': '反向雙正交小波 (Reverse Biorthogonal) 2.6', 'rbio2.8': '反向雙正交小波 (Reverse Biorthogonal) 2.8',
                'rbio3.1': '反向雙正交小波 (Reverse Biorthogonal) 3.1', 'rbio3.3': '反向雙正交小波 (Reverse Biorthogonal) 3.3', 'rbio3.5': '反向雙正交小波 (Reverse Biorthogonal) 3.5', 'rbio3.7': '反向雙正交小波 (Reverse Biorthogonal) 3.7',
                'rbio3.9': '反向雙正交小波 (Reverse Biorthogonal) 3.9', 'rbio4.4': '反向雙正交小波 (Reverse Biorthogonal) 4.4', 'rbio5.5': '反向雙正交小波 (Reverse Biorthogonal) 5.5', 'rbio6.8': '反向雙正交小波 (Reverse Biorthogonal) 6.8'
            }

            display_wavelet_options = [f"{wavelet_name_map.get(w, w)} ({w})" for w in wavelet_options]
            selected_display_wavelet = st.selectbox(
                "選擇小波類型：",
                options=display_wavelet_options,
                index=display_wavelet_options.index(f"{wavelet_name_map.get('db1', 'db1')} (db1)") if f"{wavelet_name_map.get('db1', 'db1')} (db1)" in display_wavelet_options else 0,
                key="dwt_wavelet_display"
            )

            dwt_wavelet = selected_display_wavelet.split('(')[-1][:-1]

            with st.expander("🛰️ 如何選擇小波類型？"):
                st.markdown("""
                    - **小波家族 (Family)**：不同的家族有不同的特性
                         - `db` (Daubechies) 是非對稱的正交小波
                         - `sym` (Symlets) 則是近似對稱的正交小波
                         - `bior` (Biorthogonal) 則是對稱的雙正交小波

                    - **數字/階數 (Order)**：名稱中的數字通常代表小波的「階數」
                        - **階數越高**：小波函數越平滑、支撐長度越長，有利於壓縮平滑的訊號部分
                        - **階數越低**：小波函數的局部性越好，有利於偵測訊號中的突變點或尖峰

                    - **如何選擇**：選擇哪種小波是一個權衡過程，取決於您的訊號特性，建議可以從 `db4` 或 `sym4` 開始嘗試並觀察壓縮後訊號的變化
                """)

            
            max_level = pywt.dwt_max_level(len(cleaned_df[dwt_feature]), pywt.Wavelet(dwt_wavelet)) if dwt_wavelet else 1
            dwt_level = st.slider(
                "選擇分解層級：",
                min_value=1,
                max_value=max_level,
                value=min(2, max_level),
                step=1,
                key="dwt_level"
            )

            if st.button("執行 DWT 壓縮", key="run_dwt"):
                with st.spinner("執行 DWT 壓縮中..."):
                    try:
                        original_signal, reconstructed_signal, compressed_size = preprocessor.perform_dwt_compression(
                            cleaned_df[dwt_feature], dwt_wavelet, dwt_level
                        )
                        st.markdown("##### 壓縮前後訊號比較")
                        plot_df = pd.DataFrame({
                            '原始訊號': original_signal,
                            'DWT 壓縮後訊號': reconstructed_signal
                        })

                        st.line_chart(plot_df, color=['#3CBBDE', '#F5C65D'], x_label='樣本索引', y_label=dwt_feature, height=400)
                        
                        st.markdown("##### 壓縮效果")
                        col1, col2 = st.columns(2)
                        original_size = len(original_signal)
                        compression_ratio = 1 - (compressed_size / original_size) if original_size > 0 else 0
                        col1.metric("原始資料長度", f"{original_size:,}")
                        col2.metric("壓縮後係數長度", f"{compressed_size:,}", delta=f"{compression_ratio:.2%} 壓縮率")


                        with st.expander("📊 結論：DWT (離散小波轉換) 資料壓縮"):
                            st.markdown(textwrap.dedent(f"""
                                - **精簡原理**：
                                     - DWT 將訊號分解為代表 <span style='color:#4481D7'>**「趨勢」**</span> 的**近似係數（低頻）** 和代表 <span style='color:#4481D7'>**「細節」**</span>的**細節係數（高頻）**
                                     - 此處的壓縮是透過 <span style='color:#4481D7'>**只保留最重要的近似係數**</span> 來重建訊號，進而達到壓縮目的
                                
                                - **資料特性**：
                                    - 選擇壓縮的特徵為 <span style='color:#4481D7'>**{dwt_feature}**</span>，該特徵為數值型且具有時間序列特性
                                    - 數據處理步驟：
                                        1. 訊號長度為 <span style='color:#4481D7'>**{original_size:,}**</span> 個樣本點，即原始資料需要儲存 <span style='color:#4481D7'>**{original_size:,}**</span> 個點
                                        2. 壓縮後僅保留 <span style='color:#4481D7'>**{compressed_size:,}**</span> 個近似係數，即壓縮後只需儲存 <span style='color:#4481D7'>**{compressed_size:,}**</span> 個近似係數
                                        3. 濾除掉高頻的細節係數以減少資料量，壓縮率約為 <span style='color:#4481D7'>**{compression_ratio:.2%}**</span>
                                - **方法**：
                                    - 使用 <span style='color:#4481D7'>**{wavelet_name_map.get(dwt_wavelet, dwt_wavelet)} ({dwt_wavelet})**</span> 小波，進行 <span style='color:#4481D7'>**{dwt_level}**</span> 層分解
                                - **權衡 (Trade-off)**：
                                    - 分解層級愈<span style='color:#4481D7'>**高**</span>，壓縮率愈<span style='color:#4481D7'>**高**</span>，但同時也會損失更多細節，可能導致重建的訊號過於平滑
                                - **應用場景**：
                                    - 適用於時間序列資料或訊號處理領域，如音頻壓縮、影像壓縮等
                                    """), unsafe_allow_html=True
                                    )            
                    except Exception as e:
                        st.error(f"執行 DWT 壓縮時發生錯誤: {e}")


    elif compression_method == "PCA (主成分分析)":
        st.markdown("### PCA (主成分分析) 資料壓縮")
        st.info("PCA 不僅能「降維」，還能用於「資料壓縮」，其原理是將原始資料轉換到主成分空間，然後僅保留最重要的前 k 個主成分，再將其逆轉換回原始特徵空間，此過程會損失部分資訊，但能達到壓縮資料之目的", icon="ℹ️")

        if not numerical_cols:
            st.warning("資料中沒有數值型特徵可供壓縮。", icon="⚠️")
        else:
            pca_features = st.multiselect(
                "選擇要納入 PCA 壓縮分析的數值特徵：",
                options=numerical_cols,
                default=[col for col in ['Avg_Temperature', 'Electricity_Usage'] if col in numerical_cols],
                key="pca_compression_features"
            )

            if len(pca_features) < 2:
                st.warning("請至少選擇 2 個特徵以進行有意義的 PCA 壓縮分析。", icon="⚠️")
            else:
                pca_plot_feature = st.selectbox(
                    "選擇要繪圖比較的特徵：",
                    options=pca_features,
                    index=0,
                    key="pca_plot_feature"
                )

                n_components_pca = st.slider(
                    "選擇要保留的主成分數量：",
                    min_value=1,
                    max_value=len(pca_features),
                    value=max(1, len(pca_features) // 2),
                    step=1,
                    key="pca_compression_n_components",
                    help="保留的主成分愈少，壓縮率愈高，但重建後的資訊損失也愈多。"
                )

                if st.button("執行 PCA 壓縮與重建", key="run_pca_compression"):
                    with st.spinner("執行 PCA 壓縮與重建中..."):
                        try:
                            reconstructed_df, mse, pca_model = preprocessor.perform_pca_compression(
                                pca_features, n_components_pca
                            )

                            st.markdown("##### 壓縮前後訊號比較")
                            comparison_df = pd.DataFrame({
                                '原始訊號': cleaned_df[pca_plot_feature],
                                'PCA 重建訊號': reconstructed_df[pca_plot_feature]
                            })
                            st.line_chart(comparison_df, color=['#3CBBDE', '#BC72A7'], x_label='樣本索引', y_label=pca_plot_feature, height=400)

                            st.markdown("##### 壓縮與重建評估")
                            col1, col2 = st.columns(2)
                            
                            # Compression Ratio
                            original_data_points = len(cleaned_df) * len(pca_features)
                            # PCA stores n_components * n_features (loadings) + n_components * n_samples (transformed data)
                            # Here, compressed_data_points represents the "cost" of storing the compressed representation
                            # This is a conceptual compression ratio, not byte-level compression
                            compressed_data_points = len(cleaned_df) * n_components_pca # Store transformed data
                            
                            col1.metric(
                                label="重建誤差 (MSE)", 
                                value=f"{mse:.4f}",
                                help="均方誤差 (Mean Squared Error) 用於衡量原始訊號與重建訊號之間的差異，值愈小表示重建品質愈好。"
                            )
                            
                            # Conceptual compression ratio based on reduction in dimensionality
                            # (Original dimensions - Retained dimensions) / Original dimensions
                            dimensional_reduction_ratio = (len(pca_features) - n_components_pca) / len(pca_features)
                            col2.metric(
                                label="維度精簡率", 
                                value=f"{dimensional_reduction_ratio:.2%}",
                                help=f"從 {len(pca_features)} 個原始維度精簡到 {n_components_pca} 個主成分的比例。"
                            )


                            with st.expander("📊 結論：PCA (主成分分析) 資料壓縮"):
                                st.markdown(textwrap.dedent(f"""
                                    - **精簡原理**：
                                        - PCA 找到一組新的正交基（主成分）來表示資料，這些主成分捕捉了資料中最大量的變異
                                        - 透過只保留前 `k` 個主成分，實際上是保留資料中最主要的「結構」，而忽略較次要的「噪音」或細節
                                    - **資料特性**：
                                        - 選擇納入 PCA 壓縮分析的特徵為 <span style='color:#4481D7'>**{len(pca_features)}**</span> 個數值型特徵 (`{', '.join(pca_features)}`)，這些特徵彼此之間可能存在相關性
                                        - 數據處理步驟：
                                            1. 原始資料包含 <span style='color:#4481D7'>**{len(pca_features)}**</span> 個特徵，每個特徵需儲存完整資料
                                            2. 透過 PCA 壓縮，只保留 <span style='color:#4481D7'>**{n_components_pca}**</span> 個主成分來表示這些特徵
                                            3. 重建過程中，使用這 {n_components_pca} 個 主成分來近似原始的 {len(pca_features)} 個特徵
                                        - 自上圖可知，特徵 <span style='color:#4481D7'>**`{pca_plot_feature}`**</span> 的原始值與從 {n_components_pca} 個主成分重建後的值作比較
                                        - 重建的訊號捕捉原始訊號的主要趨勢，但濾除部分波動，此差異可由 <span style='color:#4481D7'>**重建誤差 (MSE)**</span> = <span style='color:#4481D7'>**{mse:.4f}**</span> 來量化
                                    - **方法**：
                                        - 使用 PCA 模型，選擇保留 <span style='color:#4481D7'>**{n_components_pca}**</span> 個主成分來進行資料壓縮與重建
                                    - **權衡 (Trade-off)**：
                                        - 保留的主成分數量愈<span style='color:#4481D7'>**少**</span>，重建的訊號愈<span style='color:#4481D7'>**粗糙**</span>，維度精簡率愈<span style='color:#4481D7'>**高**</span> (等同於壓縮率愈<span style='color:#4481D7'>**高**</span>)，重建誤差 (MSE) 也會愈<span style='color:#4481D7'>**大**</span>，意謂著損失的資訊愈<span style='color:#4481D7'>**多**</span>
                                        - 保留的主成分數量愈<span style='color:#4481D7'>**多**</span>，重建的訊號愈<span style='color:#4481D7'>**接近**</span>原始訊號，維度精簡率愈<span style='color:#4481D7'>**低**</span>，壓縮效果也較差，重建誤差 (MSE) 也會愈<span style='color:#4481D7'>**小**</span>，意謂著保留的資訊愈<span style='color:#4481D7'>**多**</span>
                                    - **應用場景**：
                                        - 適用於需要在保留主要資訊的同時減少資料量的情境，如影像壓縮、基因資料分析等
                                        """), unsafe_allow_html=True
                                        )
                        
                        except Exception as e:
                            st.error(f"執行 PCA 壓縮時發生錯誤: {e}")