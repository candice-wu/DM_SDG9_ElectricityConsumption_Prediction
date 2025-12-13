import streamlit as st
import pandas as pd
from src.data_preprocessing import DataPreprocessor
from src.ui_components import render_app_info, render_data_status
import io

st.set_page_config(page_title="資料探索與清理", page_icon="📄", layout="wide")

def get_df_info_as_df(df):
    """將 df.info() 的輸出轉換為 DataFrame。"""
    info_df = pd.DataFrame({
        'Column': df.columns,
        'Non-Null Count': df.count().values,
        'Dtype': df.dtypes.astype(str).values # Convert dtype objects to strings
    })
    return info_df

def styled_missing_values(df):
    """回傳一個帶有樣式的缺失值 DataFrame。"""
    missing_df = df.isnull().sum().reset_index()
    missing_df.columns = ['Column', 'Count']
    def color_red(val):
        return 'color: red' if val > 0 else ''
    return missing_df.style.apply(lambda x: x.map(color_red), subset=['Count'])

def display_data_section(df, cleaned_df):
    """A helper function to display both raw and cleaned dataframes side-by-side."""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌀 原始資料集資訊")
        st.dataframe(df.style.format({'Avg_Temperature': '{:.2f}', 'Electricity_Usage': '{:.2f}'}))
        with st.expander("🔎 查看原始資料詳細資訊"):
            st.write(f"維度：{df.shape}")
            st.dataframe(get_df_info_as_df(df))
            st.write("缺失值：")
            st.dataframe(styled_missing_values(df))

    with col2:
        st.subheader("🔆 清理後的資料集資訊")
        st.dataframe(cleaned_df.style.format({'Avg_Temperature': '{:.2f}', 'Electricity_Usage': '{:.2f}'}))
        with st.expander("🔎 查看清理後資料詳細資訊"):
            st.write(f"維度：{cleaned_df.shape}")
            st.dataframe(get_df_info_as_df(cleaned_df))
            st.write("缺失值：")
            st.dataframe(styled_missing_values(cleaned_df))

def main():
    # Inject custom CSS for sidebar font size
    st.markdown("""
        <style>
            [data-testid="stSidebarNav"] a {
                font-size: 1.15rem;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Render the static info sections in the sidebar
    render_app_info()

    st.title('📄 資料探索與清理 (Data Exploration & Cleaning)')
    st.info("ℹ️ 此頁面提供上傳資料、進行資料清理，並比較清理前後的資料差異")
    st.header('上傳與清理')

    uploaded_file = st.file_uploader("請上傳您的原始資料 CSV 檔案 (或使用已上傳的資料)", type=["csv"])

    # If a new file is uploaded, process it and store in session state
    if uploaded_file is not None:
        try:
            with st.spinner("⏳ 正在處理上傳的檔案..."):
                df = pd.read_csv(uploaded_file)
                preprocessor = DataPreprocessor(df)
                cleaned_df = preprocessor.clean_data()
                
                # Store in session state
                st.session_state['df'] = df
                st.session_state['cleaned_df'] = cleaned_df
                st.session_state['preprocessor'] = preprocessor
                st.session_state['data_loaded'] = True
            st.success("檔案處理完成！")
        except Exception as e:
            st.error(f"處理檔案時發生錯誤：{e}")
            if 'data_loaded' in st.session_state:
                del st.session_state['data_loaded']

    # If data has been loaded into session state at least once, display it
    if 'data_loaded' in st.session_state and st.session_state['data_loaded']:
        # Render the dynamic data status section in the sidebar
        render_data_status(st.session_state['cleaned_df'])
        st.markdown("---")
        display_data_section(st.session_state['df'], st.session_state['cleaned_df'])

if __name__ == '__main__':
    main()