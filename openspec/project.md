# 專案背景 (Project Context)

## 專案目的與目標 (Project Purpose and Goals)
Implement an "Electricity Prediction System for Science Parks in Taiwan" project, leveraging the "Final_rawdata_20251022151713.csv" dataset. This system will utilize multiple algorithms and machine learning technologies for automated prediction, advanced data analysis, and rich data visualization. Deliverables will be presented via a Streamlit web application. The project encompasses data preprocessing, model training, visualization analysis, and web application interface development. Specific algorithms and machine learning technologies will be discussed in further detail.

基於「Final_rawdata_20251022151713.csv」資料集，並運用多種演算法和機器學習技術，實現「台灣各科技區用電量預測系統」專案。該系統將自動進行預測，並進行高階資料分析，同時添加豐富的視覺化效果，最終成果將呈現在 Streamlit 雲端網站。本項目包括資料預處理、模型訓練、視覺化分析以及Web應用程式介面開發。關於具體的演算法和機器學習技術，在後續作詳細討論或說明。

## 主要技術棧 (Key Technologies Stack)
The anticipated technologies to be utilized are listed below. Adjustments will be made as needed based on project optimization progress.

已知會運用到的技術如下，後續視專案優化程度再作調整
- Language: Python
- Data Manipulation: pandas, numpy
- Data Visualization: matplotlib, seaborn, plotly, altair
- Machine Learning: scikit-learn, lightgbm
- Scientific Computing: scipy, PyWavelets
- Web Framework: Streamlit

## 專案慣例 (Project Conventions)

### 程式碼風格 (Code Style)
- Code will be documented with comments explaining key steps.
- Adheres to the PEP 8 style guide for Python code.
- Employs meaningful variable and function names.
- Specific code style preferences: Python and related packages used in this project.
- Formatting rules: Simple and easy to understand.
- Main program naming convention: 5114050013_DM_SDG9.py

- 程式碼附有註釋以解釋關鍵步驟
- 遵循 Python 程式碼的 PEP 8 風格指南
- 使用有意義的變數名稱和函數名稱
- 程式碼風格偏好：本專案使用 Python 及相關軟體套件
- 格式化規則：簡潔易懂
- 主程式的命名為：5114050013_DM_SDG9.py

### 架構模式 (Architecture Patterns)
- The project adheres to the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology, which includes the following phases: Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, and Deployment.
- The final application will be deployed as a Streamlit web application, ensuring a clear separation between data processing, model training logic, and the user interface.

- 本專案遵循 CRISP-DM（跨產業資料探勘標準流程）方法論，該方法論包含以下階段：業務理解、資料理解、資料準備、建模、評估和部署。
- 最終應用程式以 Streamlit Web 應用程式的形式部署，將資料處理和模型訓練邏輯與使用者介面分開。

### 測試策略 (Testing Strategy)
- Unit testing will be performed initially, followed by integration testing.
- End-to-end testing can be requested separately based on project needs.
- Model performance will be evaluated using standard regression metrics: R-squared, Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).
- The Analysis Page will incorporate several visualizations for evaluation, organized into distinct tabs.

- 先單元測試，再整合測試
- end-to-end testing 視需要再另外提出請求
- 模型效能採用標準迴歸指標進行評估：R 平方值、均方根誤差 (RMSE) 和平均絕對誤差 (MAE)
- 分析頁面包括多項用於評估的視覺化圖表，並按標籤頁進行組織

### Git 工作流程 (Git Workflow)
- Upon project completion, all project files and code will be uploaded to the designated GitHub repository (DM_SDG9_ElectricityConsumption_Prediction).
- GitHub HTTPS: https://github.com/candice-wu/DM_SDG9_ElectricityConsumption_Prediction.git
- It is recommended to generate `requirements.txt` and `.gitignore` files at this stage.
- Ensure `.DS_Store` is included in the `.gitignore` file and is not pushed to the designated GitHub repository.
- The `README.md` file will outline the project's completion using the CRISP-DM architecture, with a prominent link to the Streamlit demo website showcasing the final product.
- A feature branching workflow will be employed.
- Each new feature or bug fix will be developed in a separate branch.
- Commits will be atomic, with clear and concise messages.

- 專案完成後，要把專案文件與程式碼上傳到我指定的GitHub（DM_SDG9_ElectricityConsumption_Prediction）
- GitHub HTTPS：https://github.com/candice-wu/DM_SDG9_ElectricityConsumption_Prediction.git
- 建議現在先產生requirements.txt 與 .gitignore 檔案
- 把.DS_Store 也寫進.gitignore且到時不要 push 到我指定的 GitHub
- README.md 內容是描述利用 CRISP-DM 架構來說明如何完成這項專案，最上方要帶出 Streamlit demo website 連結去呈現此專案成品
- 採用 Feature branching 工作流程
- 每個新功能或錯誤修復都在單獨的分支上開發
- 提交是 atomic，並且包含清晰簡潔的資訊

## 領域相關知識 (Domain Context)
Utilizing three distinct data sources, this project aims to predict the monthly electricity consumption trends (dependent variable) for various science parks in Taiwan. Furthermore, it will analyze the correlation between electricity consumption and factors such as monthly average temperature or other influencing variables (independent variables). The three data sources are as follows:
1. NSTC - Monthly electricity consumption data for each science park in Taiwan.
   https://wsts.nstc.gov.tw/stsweb/sciencepark/ScienceParkReport.aspx?language=C&quyid=tqelectric02
2. CWA - Climate observation station data.
   https://opendata.cwa.gov.tw/dataset/climate?page=3
3. CWA (CODiS) - Historical monthly average temperature data from various observation stations.
   https://codis.cwa.gov.tw/?utm_source=chatgpt.com

利用三個資料來源，預測台灣各科學園區在未知月份的用電量趨勢（因變數），並進一步分析用電量與月均溫或其他影響因素（自變數）之間的相關性。
三大資料來源，分列如下：
1. 國家科學及技術委員會（NSTC）提供的臺灣各個科學園區每月用電量資訊
   https://wsts.nstc.gov.tw/stsweb/sciencepark/ScienceParkReport.aspx?language=C&quyid=tqelectric02

2. 交通部中央氣象署在氣象資料開放平臺提供的氣溫測站資料
   https://opendata.cwa.gov.tw/dataset/climate?page=3

3. 交通部中央氣象署在氣候觀測資料查詢服務系統 (CODiS)提供的各氣溫測站的歷年/月之月均溫
   https://codis.cwa.gov.tw/?utm_source=chatgpt.com

## 重要限制 (Important Constraints)
- Must be capable of handling Chinese and English labels, if applicable.
- Requires memory-efficient processing for large datasets.
- Needs real-time prediction capabilities for web applications.
- Core functionality should not depend on external APIs.
- The data preprocessing sub-stage of "data cleaning" has four basic requirements (details in Chinese description below).
- The data preprocessing sub-stages also include specific requirements for "Data Integration," "Data Transformation," and "Data Reduction" (details in Chinese description below).

- 必須能夠處理中文和英文標籤（若有）
- 能夠有效率地處理大型資料集
- 具備面向 Web 應用的即時預測功能
- 核心功能無需外部 API 依賴
- 在資料預處理裡的子階段「資料清理(Data cleaning)」，有四項基本要求：
  1. 去除欄位前後空白
     df.columns = df.columns.str.strip()
  2. 補齊 Avg_Temperature 缺值
     # 規則：以「同一園區子分類」的「歷年相同月份」平均值補
     df['Avg_Temperature'] = df.groupby(['Sub_Science_Park', 'Month_NUM'])['Avg_Temperature'].transform(lambda x: x.fillna(x.mean()))

     # 若仍有補不完的（例如該園區該月全為 NaN），再用全體平均補
       df['Avg_Temperature'].fillna(df['Avg_Temperature'].mean(), inplace=True)

       print("\n✅ 補值完成，仍有缺值的筆數：", df['Avg_Temperature'].isna().sum())
  3. 確保數值型欄位型態正確
     numeric_cols = ['Year_EN', 'Month_NUM', 'Avg_Temperature', 'Electricity_Usage']
     for col in numeric_cols:
         df[col] = pd.to_numeric(df[col], errors='coerce')

  4. 類別型欄位轉換（Custom Ordering & Encoding）
     # The system now applies a custom sort order based on geographical location (North, Central, South)
     # using pandas.api.types.CategoricalDtype before encoding. This ensures visualizations
     # like box plots are displayed in a logical, geographical order.
     # 系統現在會根據預定義的地理位置（北、中、南）對分類特徵進行自定義排序，
     # 然後再進行編碼，以確保盒鬚圖等視覺化呈現有意義的順序。
     custom_orders = {
         'Science_Park': ['新竹科學園區', '中部科學園區', '南部科學園區'],
         # ... and so on for other categorical columns
     }
     for col, order in custom_orders.items():
         cat_dtype = pd.api.types.CategoricalDtype(categories=order, ordered=True)
         df[col] = df[col].astype(str).astype(cat_dtype)
         df[col] = df[col].cat.codes

- 資料預處理裡的子階段「資料整合 Data Integration」，要有相關性分析與變異數分析，以了解氣溫或北中南地區與用電量的關係強度，並有散佈圖、盒鬚圖、熱圖
- 資料預處理裡的子階段「資料轉換 Data Transformation」，我想要有以下技術去平滑資料，並讓分析結果儘量呈現常態分布，分述如下：
  一、正規化：分別利用以下三個技術做正規化，以作對比
      1. 極值正規化 Min-Max Normalization
      2. Z分配標準化 Z-score Normalization
      3. 十進位正規化 Normalization by decimal scaling
  二、離散化 Discretization：將連續型資料｜數值型資料轉換成離散型資料｜類別型資料，並套用「概念階層Concept Hierarchy
Generation」
      1. 相關性分析 Correlation (e.g., χ2) analysis
      2. 決策樹分析 Decision-tree analysis
      3. 聚類分析 Clustering analysis
  三、利用分箱法 Binning去做資料平滑化 Smoothing，且分別用以下三項去作分箱法，以作對比
      1.中位數 Smoothing by bin median
      2. 平均 Smoothing by bin means
      3. 邊界 Smoothing by bin boundaries
- 資料預處理裡的子階段「資料精簡 Data Reduction」，想要分別採用以下分類去進行
  • 維度縮減 (Dimensionality reduction)
  • 數量縮減 (Numerosity reduction)
  • 資料壓縮 (Data compression)

  更需利用以下降維技術去呈現，以作對比
  1. 離散小波轉換 Discrete Wavelet Transform，DWT
  2. 主成分分析法 PCA
  3. t-隨機鄰近嵌入法 t-distributed Stochastic Neighbor Embedding，t-SNE

  更進一步，利用「資訊理論」與「漢明距離」來進行特性的排序，並且找出對資訊量最沒有影響的（建議可以刪除的）features，帶出表格與長條圖。即「特徵排序與挑選」，除了從「資訊增益」或 SelectKBest 開始，希望再加入「漢明距離」衡量非數值(Non-numerical)的資料的相似度，接著，利用「資訊理論」來計算資訊量（亂度），即漢明算出來的機率僅為「相似度」，因此，還需加上「不相似度」帶來的資訊量（亂度）

  另外，再加入常見的幾個「距離公式」在數值型資料當中，用來求出兩個數值之間的關係
  - 歐幾里得 Euclidean
  - 曼哈頓 Manhattan
  - 切比雪夫（棋盤距離）Chebyshev
  - 敏高斯基 Minkowski

  最後，數量縮減 (Numerosity reduction) 再區分「Parametric methods (e.g., regression)」與「Non-parametric methods，Major families: histograms, clustering, sampling,…）

## 外部依賴 (External Dependencies)
- The project relies on the Final_rawdata_20251022151713.csv dataset.
- Original datasets sources:
  - NSTC - Monthly electricity consumption data for each science park in Taiwan.
    https://wsts.nstc.gov.tw/stsweb/sciencepark/ScienceParkReport.aspx?language=C&quyid=tqelectric02
  - CWA - Climate observation station data.
    https://opendata.cwa.gov.tw/dataset/climate?page=3
  - CWA (CODiS) - Historical monthly average temperature data from various observation stations.
    https://codis.cwa.gov.tw/?utm_source=chatgpt.com
- The project uses several open-source Python libraries, which are listed in `requirements.txt`.
