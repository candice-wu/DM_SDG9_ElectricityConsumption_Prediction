import streamlit as st
import pandas as pd
import datetime

def render_app_info():
    """Renders the 'About this project' and footer sections in the sidebar."""
    st.sidebar.title("關於此專案")
    st.sidebar.info(
        """
        **專案作者:** Candice Wu
        **GitHub 儲存庫:** [點此前往](https://github.com/candice-wu/DM_SDG9_ElectricityConsumption_Prediction)
        **資料來源:** 國科會、中央氣象署
        """
    )
    
    st.sidebar.divider()
    st.sidebar.caption("© 2025 Candice Wu. All Rights Reserved.")
    st.sidebar.caption(f"最後更新: {datetime.date.today().strftime('%Y-%m-%d')}")

def render_data_status(df: pd.DataFrame):
    """Renders the data status section in the sidebar if a DataFrame is provided."""
    if df is not None and not df.empty:
        # Inject custom CSS to reduce font size for metric values
        st.markdown("""
            <style>
                div[data-testid="stMetric"] > div > div {
                    font-size: 1.1rem;
                    white-space: nowrap;
                }
                div[data-testid="stMetric"] > label {
                    font-size: 1.1rem;
                }
            </style>
        """, unsafe_allow_html=True)

        st.sidebar.header("資料狀態")
        st.sidebar.metric("已載入資料筆數", f"{len(df)} 筆")
        
        # Ensure date columns exist before trying to access them
        if 'Year_EN' in df.columns and 'Month_NUM' in df.columns:
            min_year = df['Year_EN'].min()
            max_year = df['Year_EN'].max()
            # Get min month for the min year, and max month for the max year
            min_month = df[df['Year_EN'] == min_year]['Month_NUM'].min()
            max_month = df[df['Year_EN'] == max_year]['Month_NUM'].max()
            
            st.sidebar.metric("資料時間範圍", f"{min_year}/{min_month} - {max_year}/{max_month}")
