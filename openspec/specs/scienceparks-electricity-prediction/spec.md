### Requirement: Science Parks Electricity Prediction System
The system SHALL predict electricity consumption trends for science parks in Taiwan based on the `Final_rawdata_20251022151713.csv` dataset, provide advanced data analysis and rich visualizations, and ultimately offer a user interface via a Streamlit website.
系統應能基於 `Final_rawdata_20251022151713.csv`資料集，自動預測台灣各科技園區的用電量趨勢，並提供進階資料分析與豐富的視覺化呈現，最終透過 Streamlit 網站提供使用者介面。

#### Scenario: 系統初始化與資料載入
- **WHEN** 使用者在「資料探索與清理」頁面上傳 CSV 檔案
- **THEN** 系統應能成功讀取並處理資料，並將原始與清理後的 DataFrame 存入 session state。

### Requirement: Data Preprocessing Module
The Data Preprocessing Module SHALL execute according to the project-defined requirements for "Data Cleaning," "Data Integration," "Data Transformation," and "Data Reduction."
資料預處理模組應能按照專案定義的「資料清理」、「資料整合」、「資料轉換」和「資料精簡」要求執行。

#### Scenario: 資料清理
- **WHEN** 執行資料清理步驟
- **THEN** 欄位前後空白應被去除，`Avg_Temperature` 缺值應被補齊，數值型欄位型態應正確。

#### Scenario: 自定義類別排序與編碼
- **WHEN** 對地理位置相關的類別型欄位 (`Science_Park`, `Sub_Science_Park`, `County`, `Town`) 進行編碼
- **THEN** 系統應能按照預設的北中南地理區域順序進行排序和編碼，以確保視覺化的一致性與可讀性。

#### Scenario: 資料整合與視覺化分析
- **WHEN** 使用者導覽至「資料分析」頁面
- **THEN** 系統應提供以下分析圖表：
    - **特徵相關性熱圖 (Annotated Heatmap)**：顯示所有數值型特徵（包括編碼後的類別特徵）與用電量的相關性。
    - **線性迴歸圖 (Linear Regression Plot)**：展示月均溫與用電量的線性關係及邊際分佈。
    - **殘差圖 (Residual Plot)**：顯示月均溫與用電量殘差，以評估模型擬合度。
    - **盒鬚圖 (Box Plot)**：針對各類別特徵與用電量繪製盒鬚圖，X軸標籤應顯示原始中文名稱。
- **AND** 每個圖表區塊應有明確的標題（包含英文翻譯）、描述文字、以及可展開的結論區塊，結論中應突出顯示關鍵資訊。

#### Scenario: 關聯規則探勘 (Association Rule Mining)
- **WHEN** 使用者導覽至「資料分析」頁面的「關聯規則分析」分頁
- **AND** 選擇一或多個屬性特徵進行分析（包括類別型及已離散化的連續型特徵）
- **AND** 設定「最低支持度 (Min Support)」和「最小信賴度 (Min Confidence)」閾值
- **THEN** 系統應使用 Apriori 演算法，找出符合條件的高頻項目集與關聯規則
- **AND** 以表格形式顯示這些關聯規則，包含 `Antecedents` (前項), `Consequents` (後項), `Support` (支持度), `Confidence` (信賴度), 和 `Lift` (增益) 等指標。
- **AND** 提供解釋性文字，說明關聯規則的解讀方式，特別是 Lift 值 (>1, <1, =1 的意義)。

#### Scenario: 資料轉換
- **WHEN** 使用者導覽至「資料轉換」頁面
- **THEN** 系統應提供互動式介面，展示原始資料概覽、多種正規化方法（極值正規化、Z分配標準化、十進位正規化）、離散化方法（等寬/等深裝箱、決策樹、聚類、卡方檢定）及資料平滑化方法（依分箱平均值/中位數/邊界）的效果。

#### Scenario: 資料精簡 - 維度縮減 (視覺化)
- **WHEN** 使用者在「資料精簡」頁面的「維度縮減」分頁中，選擇 PCA 或 t-SNE 降維方法並執行
- **THEN** 系統應顯示對應的碎石圖（僅PCA）、2D/3D散佈圖，以及包含方法定義與數據特性的結論。

#### Scenario: 資料精簡 - 維度縮減 (特徵排序與距離度量)
- **WHEN** 使用者在「資料精簡」頁面的「維度縮減」分頁的「特徵排序與距離度量」子分頁中進行操作
- **THEN** 系統應能計算並展示數值型與類別型特徵的重要性排序，並提供互動式介面來計算不同資料點之間的數值距離（歐幾里得、曼哈頓等）和類別距離（漢明距離）。

#### Scenario: 資料精簡 - 數量縮減
- **WHEN** 使用者在「資料精簡」頁面的「數量縮減」分頁中，選擇參數方法（迴歸）、非參數方法（直方圖、叢集）或抽樣方法
- **THEN** 系統應顯示對應的圖表與摘要（如迴歸線、直方圖、叢集質心、樣本分佈比較），以證明數據量被有效縮減，並解釋其原理與權衡。

#### Scenario: 資料精簡 - 資料壓縮
- **WHEN** 使用者在「資料精簡」頁面的「資料壓縮」分頁中，選擇 DWT 或 PCA 壓縮方法並執行
- **THEN** 系統應顯示原始與重建後的數據對比圖，並提供壓縮率或重建誤差等指標，展示壓縮效果。

### Requirement: Model Training Module
The Model Training Module SHALL implement multiple machine learning algorithms for electricity consumption prediction, allowing for comparison and evaluation.
模型訓練模組應實現多種機器學習演算法進行用電量預測，並允許比較和評估。

#### Scenario: 多模型預測與評估
- **WHEN** 使用者在「用電量預測」頁面的下拉選單中選擇一個模型（例如「梯度提升樹迴歸 (HistGradient)」）、設定超參數並點擊「預估」
- **THEN** 系統應使用所選模型及超參數產生預測值
- **AND** 在頁面上顯示該模型的效能指標（R-squared, RMSE, MAE）。

#### Scenario: 模型訓練與預測
- **WHEN** 使用者在「用電量預測」頁面點擊「預估」按鈕
- **THEN** 系統應能使用已快取的機器學習模型，根據側邊欄的輸入特徵，輸出用電量預測值。

### Requirement: Visualization Module
The Visualization Module SHALL provide diverse charts and dashboards to display prediction results, model performance evaluation, and the correlation between electricity consumption and influencing factors.
視覺化模組應提供多樣化的圖表和儀表板，展示預測結果、模型效能評估以及用電量與影響因素之間的相關性。

#### Scenario: 視覺化呈現
- **WHEN** 請求視覺化報告
- **THEN** 系統應能生成多樣化的圖表，展示預測結果、模型效能評估（R平方值、RMSE、MAE）以及用電量與影響因素之間的相關性。

#### Scenario: 進階模型比較
- **WHEN** 使用者導覽至「模型比較分析」頁面並選擇一個或多個模型
- **THEN** 系統應顯示「預測值 vs. 實際值」散佈圖和「殘差圖」，以視覺化方式比較所選模型的效能與行為，並提供詳細的結論區塊。
- **AND** 當只選擇一個模型時，系統應在獨立欄位中顯示該模型的「特徵重要性」長條圖，支援線性迴歸、決策樹、LightGBM、HistGradient（使用排列重要性計算）等模型。

#### Scenario: 級距預測準確度評估
- **WHEN** 使用者在「模型比較分析」頁面選擇一個模型並定義用電量級距（例如：低、中、高）
- **THEN** 系統應將模型的連續預測值與實際值轉換為對應級距，並顯示一個視覺化的混淆矩陣 (Confusion Matrix)。
- **AND** 混淆矩陣的軸標籤應以易於理解的方式呈現（例如：「低用量 (min-max]」），並正確處理「預測超出範圍」的預測值。
- **AND** 結論區塊應提供整體準確度、對角線/非對角線數值的意義，以及對「預測超出範圍」情況的詳細說明。

### Requirement: Streamlit Web Application
The Streamlit web application SHALL be successfully deployed, providing an interactive, multi-page user interface to upload data, configure predictions, and view results.
Streamlit 網站應能成功部署，並提供一個互動式、多頁面的使用者介面，用於上傳資料、設定預測參數並檢視結果。

#### Scenario: 多頁面導覽與狀態保存
- **WHEN** 使用者在「資料探索與清理」頁面上傳資料後，切換至「用電量預測」頁面
- **THEN** 「用電量預測」頁面應能直接使用已清理的資料，且使用者在側邊欄的設定應在頁面刷新後被保留。

#### Scenario: 預測未來日期
- **WHEN** 使用者選擇一個資料集中不存在的未來「西元年」或「月份」
- **THEN** 系統應在側邊欄的「月均溫」輸入框中，自動填入該子園區歷年同月份的平均溫度作為智慧預設值，並允許使用者手動修改。

#### Scenario: 預測回饋機制
- **WHEN** 系統產生預測值，且使用者提供了實際用電量
- **THEN** 系統應計算兩者差異百分比，並顯示一個半月型儀表板視覺化誤差程度。
- **AND** 根據以下規則顯示回饋文字：
    - 誤差為 0% 時，顯示「That's awesome! 預測完全準確！」並觸發氣球特效。
    - 誤差率 ≤ 3% 時，顯示「😊 誤差率低於 3%，太棒了！感謝您的正面回饋」。
    - 3% < 誤差率 ≤ 5% 時，顯示「😐 誤差率介於 3% ~ 5%，感謝您的回饋，我們會繼續努力👍」。
    - 誤差率 > 5% 時，顯示「😥 Oh no！誤差率高於 5%，感謝您的回饋，我們會參考這項資訊來改進模型」。

#### Scenario: 清除預測狀態
- **WHEN** 使用者點擊「清除」按鈕
- **THEN** 所有側邊欄的輸入應重設為預設值，且主畫面上的預測結果應被清除。

### Requirement: Code Quality and Conventions
The code SHALL adhere to the PEP 8 style guide and include necessary comments.
程式碼應遵循 PEP 8 風格指南，並包含必要的註釋。

#### Scenario: 程式碼風格檢查
- **WHEN** 執行程式碼風格檢查
- **THEN** 程式碼應符合 PEP 8 規範，並包含解釋關鍵步驟的註釋。
