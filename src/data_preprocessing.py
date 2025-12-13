import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor # Changed from RandomForestRegressor
from sklearn.svm import SVR # Added
from sklearn.model_selection import train_test_split # Added
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error # mean_absolute_error Added
from scipy.spatial.distance import euclidean, cityblock, chebyshev, minkowski, hamming
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import numpy as np
import matplotlib.font_manager as fm
import matplotlib.lines as mlines
import scipy.stats
import pywt
import lightgbm as lgb # Added for LightGBM

class DataPreprocessor:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.encoders = {}
        self.scaler = None # Initialize scaler to None

    def clean_data(self):
        # 1. 去除欄位前後空白
        self.df.columns = self.df.columns.str.strip()

        # 2. 補齊 Avg_Temperature 缺值
        # 規則：以「同一園區子分類」的「歷年相同月份」平均值補
        self.df['Avg_Temperature'] = self.df.groupby(['Sub_Science_Park', 'Month_NUM'])['Avg_Temperature'].transform(lambda x: x.fillna(x.mean()))
        # 若仍有補不完的（例如該園區該月全為 NaN），再用全體平均補
        self.df['Avg_Temperature'] = self.df['Avg_Temperature'].fillna(self.df['Avg_Temperature'].mean())

        # 3. 確保數值型欄位型態正確
        numeric_cols = ['Year_EN', 'Month_NUM', 'Avg_Temperature', 'Electricity_Usage']
        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # 4. 類別型欄位轉換（Custom Ordering & Encoding）
        custom_orders = {
            'Science_Park': ['新竹科學園區', '中部科學園區', '南部科學園區'],
            'Sub_Science_Park': ['新竹園區', '新竹生醫園區', '竹南園區', '龍潭園區', '銅鑼園區', '宜蘭園區', 
                                 '台中園區', '后里園區', '二林園區', '中興園區', '虎尾園區', '臺南園區', '高雄園區'],
            'County': ['宜蘭縣', '桃園市', '新竹市', '新竹縣', '苗栗縣', '臺中市', '彰化縣', '南投縣', '雲林縣', '臺南市', '高雄市'],
            'Town': ['宜蘭市', '龍潭區、平鎮區、楊梅區', '新竹市', '竹北市', '竹南市', '銅鑼鄉', '西屯區', 
                     '后里區', '二林鎮', '南投市', '虎尾鎮', '新市、善化、安定', '路竹區']
        }

        for col, order in custom_orders.items():
            cat_dtype = pd.api.types.CategoricalDtype(categories=order, ordered=True)
            self.df[col] = self.df[col].astype(str).astype(cat_dtype)
            self.encoders[col] = order
            self.df[col] = self.df[col].cat.codes
        
        # Cast encoded columns to int64 to ensure they are included in correlation matrix
        for col in custom_orders.keys():
            self.df[col] = self.df[col].astype('int64')

        return self.df

    def integrate_data(self):
        # Set Chinese font for matplotlib
        try:
            font_path = 'fonts/Noto_Sans_TC/NotoSansTC-VariableFont_wght.ttf'
            font_name = "Noto Sans TC"

            # Check if the font is already registered, if not, add it and rebuild cache
            if font_name not in [f.name for f in fm.fontManager.ttflist]:
                fm.fontManager.addfont(font_path)
                fm._load_fontmanager(try_read_cache=False)  # Force cache rebuild

            plt.rcParams['font.family'] = font_name
            plt.rcParams['axes.unicode_minus'] = False
        except Exception as e:
            print(f"Cannot set Chinese font: {e}")

        plots = {}
        analysis_results = {'box_plot_analysis': {}}

        # Correlation Analysis & Heatmap
        numerical_df = self.df.select_dtypes(include=np.number)
        correlation_matrix = numerical_df.corr()
        
        f, ax = plt.subplots(figsize=(11, 9))
        cmap = sns.color_palette("YlOrBr", as_cmap=True)
        sns.heatmap(correlation_matrix, cmap=cmap, annot=True, fmt=".2f", linewidths=.5, linecolor='black')
        plt.title('相關性熱圖 (Correlation Heatmap)', fontsize=16)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(f)
        plots['correlation_heatmap'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        analysis_results['correlation_matrix'] = correlation_matrix

        # Linear regression with marginal distributions
        # Perform linear regression to get statistics
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(self.df['Avg_Temperature'], self.df['Electricity_Usage'])
        analysis_results['linear_regression_stats'] = {
            'slope': slope,
            'intercept': intercept,
            'r_value': r_value,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_err': std_err
        }

        g = sns.jointplot(x="Avg_Temperature", y="Electricity_Usage", data=self.df, kind="reg",
                          joint_kws={'line_kws': {'color': '#DD6D6A'}, 'scatter_kws': {'color': '#739DBC', 'alpha': 0.5}},
                          marginal_kws={'color': '#9372A7'})
        g.fig.suptitle('線性迴歸暨邊際分佈圖')
        g.set_axis_labels("月均溫 (°C)", "用電量 (萬KW)")
        
        legend_elements_joint = [mlines.Line2D([0], [0], marker='o', color='w', label='資料點',
                                          markerfacecolor='#739DBC', markersize=10),
                                  mlines.Line2D([0], [0], color='#DD6D6A', lw=2, label='迴歸線')]
        g.ax_joint.legend(handles=legend_elements_joint)
        
        g.fig.tight_layout()
        buf = io.BytesIO()
        g.savefig(buf, format='png')
        plt.close(g.fig)
        plots['jointplot_reg'] = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Residual plot: Avg_Temperature vs Electricity_Usage
        fig_resid, ax_resid = plt.subplots(figsize=(10, 6))
        sns.residplot(x='Avg_Temperature', y='Electricity_Usage', data=self.df, lowess=True, 
                      ax=ax_resid,
                      scatter_kws={'color': '#7FA550', 'alpha': 0.5}, 
                      line_kws={'color': '#DD6D6A', 'linestyle': '-', 'linewidth': 2})
        ax_resid.axhline(0, color='#739DBC', linestyle='--')
        
        legend_elements = [mlines.Line2D([0], [0], marker='o', color='w', label='殘差',
                                          markerfacecolor='#7FA550', markersize=10),
                           mlines.Line2D([0], [0], color='#DD6D6A', lw=2, label='趨勢線'),
                           mlines.Line2D([0], [0], color='#739DBC', lw=2, linestyle='--', label='零線')]
        ax_resid.legend(handles=legend_elements)

        plt.title('月均溫與用電量殘差圖', fontsize=16)
        plt.xlabel('月均溫 (°C)')
        plt.ylabel('殘差 (Residuals)')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig_resid)
        plots['residual_temp_vs_elec'] = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Box plots for categorical features vs Electricity_Usage
        categorical_cols = ['Science_Park', 'Sub_Science_Park', 'County', 'Town']
        boxplot_colors = {
            'Science_Park': '#3CBBDE',
            'Sub_Science_Park': '#9FCE63',
            'County': '#DD6D6A',
            'Town': '#F5C65D'
        }
        
        for col in categorical_cols:
            try:
                fig_box, ax_box = plt.subplots(figsize=(12, 6))
                sns.boxplot(x=col, y='Electricity_Usage', data=self.df, ax=ax_box, color=boxplot_colors.get(col))
                
                if col in self.encoders:
                    category_list = self.encoders[col]
                    ticks = [tick for tick in ax_box.get_xticks() if tick < len(category_list)]
                    labels = [category_list[int(tick)] for tick in ticks]
                    ax_box.set_xticklabels(labels, rotation=45, ha='right')

                ax_box.set_title(f'{col} 與用電量盒鬚圖', fontsize=16)
                ax_box.set_xlabel(col)
                ax_box.set_ylabel('用電量 (萬KW)')
                
                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                plt.close(fig_box)
                plots[f'boxplot_{col}_vs_elec'] = base64.b64encode(buf.getvalue()).decode('utf-8')

                median_data = self.df.groupby(col)['Electricity_Usage'].median().sort_values(ascending=False)
                if col in self.encoders:
                    original_labels = {i: label for i, label in enumerate(self.encoders[col])}
                    median_data.index = median_data.index.map(original_labels)
                analysis_results['box_plot_analysis'][col] = median_data
            except Exception as e:
                print(f"Error generating boxplot for {col}: {e}")
                plots[f'boxplot_{col}_vs_elec'] = None # Ensure key exists but is empty

        return plots, analysis_results

    def transform_data(self):
        """
        Applies all normalization techniques to the numerical columns.
        """
        df_copy = self.df.copy()
        columns_to_normalize = ['Avg_Temperature', 'Electricity_Usage']

        for col in columns_to_normalize:
            # Min-Max Normalization
            min_max_scaler = MinMaxScaler()
            df_copy[f"{col}_min_max"] = min_max_scaler.fit_transform(df_copy[[col]])

            # Z-score Normalization
            z_score_scaler = StandardScaler()
            df_copy[f"{col}_z_score"] = z_score_scaler.fit_transform(df_copy[[col]])

            # Decimal Scaling Normalization
            p = df_copy[col].abs().max()
            q = 0
            if p > 0:
                while p >= 1:
                    p = p/10
                    q += 1
            df_copy[f"{col}_decimal_scaled"] = df_copy[col] / (10**q)
            
        return df_copy

    def reduce_data(self):
        # TODO: Implement data reduction steps
        pass

    def get_raw_data_statistics(self, column):
        """Calculates a comprehensive set of statistics for a given column."""
        stats = {}
        data = self.df[column].dropna()

        # Basic statistics
        stats['Mean'] = data.mean()
        stats['Median'] = data.median()
        stats['Mode'] = data.mode().iloc[0] if not data.mode().empty else 'N/A'
        
        # Five-number summary
        five_num = data.describe(percentiles=[.25, .5, .75])
        stats['Min'] = five_num['min']
        stats['Q1'] = five_num['25%']
        stats['Q2 (Median)'] = five_num['50%']
        stats['Q3'] = five_num['75%']
        stats['Max'] = five_num['max']
        
        # Dispersion
        stats['Variance'] = data.var()
        stats['Standard Deviation'] = data.std()
        
        # Outlier detection using IQR method
        Q1 = stats['Q1']
        Q3 = stats['Q3']
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        stats['Outliers'] = len(outliers)
        
        return stats

    def apply_equal_width_binning(self, column, bins=5, labels=None):
        """
        Applies equal-width binning to a specified column.
        :param column: The name of the column to discretize.
        :param bins: The number of bins to create.
        :param labels: Optional labels for the bins. If None, uses default interval labels.
        :return: A Series containing the discretized data.
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")
        
        # Ensure the column is numeric for binning
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            raise TypeError(f"Column '{column}' is not numeric and cannot be binned.")

        # Perform equal-width binning
        discretized_series = pd.cut(self.df[column], bins=bins, labels=labels, include_lowest=True, right=True)
        return discretized_series

    def apply_equal_depth_binning(self, column, bins=5, labels=None):
        """
        Applies equal-depth (quantile) binning to a specified column.
        :param column: The name of the column to discretize.
        :param bins: The number of bins to create.
        :param labels: Optional labels for the bins. If None, uses default interval labels.
        :return: A Series containing the discretized data.
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")
        
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            raise TypeError(f"Column '{column}' is not numeric and cannot be binned.")

        # Perform equal-depth binning, handling non-unique bin edges
        discretized_series = pd.qcut(self.df[column], q=bins, labels=labels, duplicates='drop')
        return discretized_series

    def apply_decision_tree_binning(self, feature_col, target_col='Electricity_Usage', max_depth=3):
        """
        Applies decision tree-based binning to a specified feature column.
        :param feature_col: The name of the continuous feature column to discretize.
        :param target_col: The name of the continuous target column.
        :param max_depth: The maximum depth of the decision tree, which influences the number of bins.
        :return: A Series containing the discretized data.
        """
        if feature_col not in self.df.columns:
            raise ValueError(f"Feature column '{feature_col}' not found in the DataFrame.")
        if target_col not in self.df.columns:
            raise ValueError(f"Target column '{target_col}' not found in the DataFrame.")
        if feature_col == target_col:
            raise ValueError("Feature and target columns cannot be the same.")

        # 1. Discretize the continuous target variable into 3 bins for classification
        target_binned = pd.qcut(self.df[target_col], q=4, labels=False, duplicates='drop')
        
        # 2. Train a Decision Tree Classifier
        tree_model = DecisionTreeClassifier(max_depth=max_depth)
        # Reshape X to be a 2D array
        X = self.df[[feature_col]].values
        tree_model.fit(X, target_binned)
        
        # 3. Extract thresholds from the trained tree
        # The thresholds are the split points at the internal nodes
        thresholds = sorted([t for t in tree_model.tree_.threshold if t != -2])
        
        if not thresholds:
            raise RuntimeError("Decision tree could not find any split points. Try a different feature or max_depth.")

        # 4. Create bins using the extracted thresholds
        # Add infinity to cover the full range of data
        bins = [-np.inf] + thresholds + [np.inf]
        
        # 5. Apply the bins to the original feature column
        discretized_series = pd.cut(self.df[feature_col], bins=bins, right=True, include_lowest=True)
        
        return discretized_series

    def apply_clustering_binning(self, feature_col, n_clusters=5):
        """
        Applies clustering-based (K-Means) binning to a specified feature column.
        :param feature_col: The name of the continuous feature column to discretize.
        :param n_clusters: The number of clusters (bins) to create.
        :return: A Series containing the discretized data with interval labels.
        """
        if feature_col not in self.df.columns:
            raise ValueError(f"Feature column '{feature_col}' not found in the DataFrame.")
        if not pd.api.types.is_numeric_dtype(self.df[feature_col]):
            raise TypeError(f"Column '{feature_col}' is not numeric and cannot be binned.")

        # 1. Fit K-Means model
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        X = self.df[[feature_col]].values
        cluster_labels = kmeans.fit_predict(X)
        
        # Create a temporary DataFrame to work with
        temp_df = pd.DataFrame({
            'feature': self.df[feature_col],
            'cluster': cluster_labels
        })
        
        # 2. Create interpretable interval labels for each cluster
        cluster_ranges = temp_df.groupby('cluster')['feature'].agg(['min', 'max'])
        # Sort cluster ranges by the min value of the feature
        cluster_ranges = cluster_ranges.sort_values(by='min')
        
        # Create labels like (min-max]
        cluster_ranges['label'] = cluster_ranges.apply(lambda row: f"({row['min']:.2f} - {row['max']:.2f}]", axis=1)
        
        # 3. Map cluster labels back to the new interval labels
        label_map = cluster_ranges['label'].to_dict()
        
        # Create a mapping from the original kmeans label to the sorted label order
        sorted_cluster_map = {original_label: new_label for new_label, original_label in enumerate(cluster_ranges.index)}
        
        # First map to the sorted order, then map to the interval string
        sorted_labels = temp_df['cluster'].map(sorted_cluster_map)
        final_labels = sorted_labels.map(label_map)

        # 4. Convert the final labels to a categorical type to maintain order in plots
        discretized_categorical = pd.Categorical(final_labels, categories=cluster_ranges['label'], ordered=True)
        discretized_series = pd.Series(discretized_categorical)

        return discretized_series

    def smooth_by_bin_mean(self, column, bins=5):
        """
        透過分箱平均值平滑資料
        """
        if column not in self.df.columns:
            raise ValueError(f"欄位 '{column}' 不存在")
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            raise TypeError(f"欄位 '{column}' 必須是數值型")

        binned_data = pd.cut(self.df[column], bins=bins, include_lowest=True, right=True)
        bin_means = self.df[column].groupby(binned_data).transform('mean')
        return bin_means

    def smooth_by_bin_median(self, column, bins=5):
        """
        透過分箱中位數平滑資料
        """
        if column not in self.df.columns:
            raise ValueError(f"欄位 '{column}' 不存在")
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            raise TypeError(f"欄位 '{column}' 必須是數值型")

        binned_data = pd.cut(self.df[column], bins=bins, include_lowest=True, right=True)
        bin_medians = self.df[column].groupby(binned_data).transform('median')
        return bin_medians

    def smooth_by_bin_boundaries(self, column, bins=5):
        """
        透過最接近的分箱邊界平滑資料
        """
        if column not in self.df.columns:
            raise ValueError(f"欄位 '{column}' 不存在")
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            raise TypeError(f"欄位 '{column}' 必須是數值型")

        # 建立分箱並取得邊界
        binned_series, bin_edges = pd.cut(self.df[column], bins=bins, include_lowest=True, right=True, retbins=True)
        
        # 對於每個值，找到最接近的邊界
        def find_closest_boundary(value):
            closest_boundary = None
            min_diff = float('inf')
            for edge in bin_edges:
                diff = abs(value - edge)
                if diff < min_diff:
                    min_diff = diff
                    closest_boundary = edge
            return closest_boundary

        return self.df[column].apply(find_closest_boundary)

    def apply_pca(self, features, n_components):
        """
        Applies PCA to the specified features.
        """
        if not all(f in self.df.columns for f in features):
            raise ValueError("指定的特徵中，有一個或多個不存在")
            
        # Standardize the features before applying PCA
        X = self.df[features].values
        X_scaled = StandardScaler().fit_transform(X)
        
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(X_scaled)
        
        # Create a new DataFrame with the principal components
        pca_df = pd.DataFrame(data=principal_components, 
                              columns=[f'PC_{i+1}' for i in range(n_components)])
                              
        return pca, pca_df

    def apply_tsne(self, features, perplexity, learning_rate):
        """
        Applies t-SNE to the specified features.
        """
        if not all(f in self.df.columns for f in features):
            raise ValueError("指定的特徵中，有一個或多個不存在")

        # Standardize the features before applying t-SNE
        X = self.df[features].values
        X_scaled = StandardScaler().fit_transform(X)

        tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=42)
        tsne_results = tsne.fit_transform(X_scaled)

        # Create a new DataFrame with the t-SNE components
        tsne_df = pd.DataFrame(data=tsne_results, columns=['t-SNE 1', 't-SNE 2'])
        
        return tsne_df

    def calculate_mutual_info(self, features, target_col):
        """
        Calculates the mutual information between numerical features and a target column.
        """
        if not all(f in self.df.columns for f in features):
            raise ValueError("指定的特徵中，有一個或多個不存在")
        if target_col not in self.df.columns:
            raise ValueError(f"目標欄位 '{target_col}' 不存在")

        X = self.df[features]
        y = self.df[target_col]
        
        mi_scores = mutual_info_regression(X, y, random_state=42)
        mi_series = pd.Series(mi_scores, index=X.columns)
        mi_series = mi_series.sort_values(ascending=False)
        
        return mi_series
    
    def calculate_information_gain(self, features, target_col, bins=5):
        """
        Calculates the information gain (mutual information for classification) between numerical
        features and a discretized target column.
        """
        if not all(f in self.df.columns for f in features):
            raise ValueError("指定的特徵中，有一個或多個不存在")
        if target_col not in self.df.columns:
            raise ValueError(f"目標欄位 '{target_col}' 不存在")

        X = self.df[features]
        # Discretize the continuous target variable to calculate information gain
        y_binned = pd.qcut(self.df[target_col], q=4, labels=False, duplicates='drop')
        
        ig_scores = mutual_info_classif(X, y_binned, random_state=42)
        ig_series = pd.Series(ig_scores, index=X.columns)
        ig_series = ig_series.sort_values(ascending=False)
        
        return ig_series

    def calculate_distance_metrics(self, index1, index2, features, p_minkowski=3):
        """
        Calculates various distance metrics between two data points.
        """
        if index1 not in self.df.index or index2 not in self.df.index:
            raise ValueError("提供的索引至少有一個超出範圍")
        if not all(f in self.df.columns for f in features):
            raise ValueError("指定的特徵中，有一個或多個不存在")

        # Extract the two data points (vectors)
        point1 = self.df.loc[index1, features].values
        point2 = self.df.loc[index2, features].values
        
        # Calculate distances
        distances = {
            'Euclidean': euclidean(point1, point2),
            'Manhattan': cityblock(point1, point2),
            'Chebyshev': chebyshev(point1, point2),
            'Minkowski': minkowski(point1, point2, p=p_minkowski)
        }
        
        return distances
    def calculate_hamming_distance(self, index1, index2, features):
        """
        Calculates the Hamming distance between two data points for categorical features.
        """
        if index1 not in self.df.index or index2 not in self.df.index:
            raise ValueError("提供的索引至少有一個超出範圍")
        if not all(f in self.df.columns for f in features):
            raise ValueError("指定的特徵中，有一個或多個不存在")

        # Extract the two data points (vectors)
        point1 = self.df.loc[index1, features].values
        point2 = self.df.loc[index2, features].values
        
        # The hamming function from scipy returns the normalized distance (proportion of mismatches).
        # We multiply by the number of features to get the raw mismatch count.
        mismatch_count = hamming(point1, point2) * len(features)
        
        return {
            'Hamming Distance': int(mismatch_count),
            'Compared Features': len(features)
        }
    
    def perform_linear_regression(self, feature_col, target_col):
        """
        Performs linear regression between a feature column (X) and a target column (Y).
        Returns the model, coefficients, intercept, and R-squared score.
        """
        if feature_col not in self.df.columns:
            raise ValueError(f"Feature column '{feature_col}' not found in the DataFrame.")
        if target_col not in self.df.columns:
            raise ValueError(f"Target column '{target_col}' not found in the DataFrame.")
        
        X = self.df[[feature_col]].values
        y = self.df[target_col].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        y_pred = model.predict(X)
        r_squared = r2_score(y, y_pred)
        
        return {
            'model': model,
            'coefficient': model.coef_[0],
            'intercept': model.intercept_,
            'r_squared': r_squared
        }
    
    def perform_decision_tree_regression(self, feature_col, target_col, max_depth=5, random_state=42):
        """
        Performs decision tree regression between a feature column (X) and a target column (Y).
        Returns the trained model and R-squared score.
        """
        if feature_col not in self.df.columns:
            raise ValueError(f"Feature column '{feature_col}' not found in the DataFrame.")
        if target_col not in self.df.columns:
            raise ValueError(f"Target column '{target_col}' not found in the DataFrame.")
        
        X = self.df[[feature_col]].values
        y = self.df[target_col].values
        
        model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
        model.fit(X, y)
        
        y_pred = model.predict(X)
        r_squared = r2_score(y, y_pred)
        
        return {
            'model': model,
            'r_squared': r_squared
        }

    def create_histogram_bins(self, feature_col, bins):
        """
        Creates histogram bins and counts for a given feature.
        """
        if feature_col not in self.df.columns:
            raise ValueError(f"Feature column '{feature_col}' not found in the DataFrame.")
        if not pd.api.types.is_numeric_dtype(self.df[feature_col]):
            raise TypeError(f"Column '{feature_col}' is not numeric and cannot be binned.")

        binned_data = pd.cut(self.df[feature_col], bins=bins, include_lowest=True, right=True)
        histogram_counts = binned_data.value_counts().sort_index()
        
        # Prepare data for st.bar_chart which expects a DataFrame with a specific format
        # The index should be the category, and columns are the values.
        # Let's convert intervals to strings for cleaner labels
        ordered_labels = [str(interval) for interval in histogram_counts.index]
        histogram_df = pd.DataFrame({'count': histogram_counts.values}, index=pd.CategoricalIndex(ordered_labels, categories=ordered_labels, ordered=True))
        
        return histogram_df
        
    def perform_clustering_reduction(self, features, n_clusters):
        """
        Performs K-Means clustering for numerosity reduction.
        """
        if not all(f in self.df.columns for f in features):
            raise ValueError("指定的特徵中，有一個或多個不存在。")
        if not all(pd.api.types.is_numeric_dtype(self.df[f]) for f in features):
            raise TypeError("所有用於群集的特徵都必須是數值型。")

        X = self.df[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Get centroids and inverse transform them to original scale
        centroids_scaled = kmeans.cluster_centers_
        centroids = scaler.inverse_transform(centroids_scaled)
        
        centroids_df = pd.DataFrame(centroids, columns=features)
        
        # Add a 'Cluster' column for easier identification and sorting in UI
        centroids_df['Cluster'] = [f"Cluster {i+1}" for i in range(n_clusters)]
        
        # Sort by the new 'Cluster' column
        centroids_df = centroids_df.sort_values(by='Cluster').reset_index(drop=True)

        # Create descriptive labels for each cluster based on centroid values
        # This will be used for legend in UI
        descriptive_labels = {}
        for i in range(n_clusters):
            label_parts = [f"Cluster {i+1}"]
            for feature in features:
                centroid_val = centroids_df.loc[i, feature]
                label_parts.append(f"{feature}: {centroid_val:.2f}")
            descriptive_labels[i] = " (".join(label_parts) + ")"
        
        return {
            'cluster_labels': cluster_labels,
            'cluster_centroids': centroids_df,
            'descriptive_labels': descriptive_labels
        }

    def perform_random_sampling(self, sample_frac):
        """
        Performs simple random sampling on the dataframe.
        """
        return self.df.sample(frac=sample_frac, random_state=42)

    def perform_stratified_sampling(self, sample_frac, stratify_by):
        """
        Performs stratified sampling on the dataframe.
        """
        if stratify_by not in self.df.columns:
            raise ValueError(f"Stratification column '{stratify_by}' not found.")
        
        # Using groupby and sample preserves the distribution of the 'stratify_by' column
        return self.df.groupby(stratify_by, group_keys=False).apply(lambda x: x.sample(frac=sample_frac, random_state=42))

    def perform_systematic_sampling(self, n):
        """
        Performs systematic sampling on the dataframe.
        Selects every Nth row.
        """
        if n <= 0:
            raise ValueError("The step 'n' for systematic sampling must be positive.")
        if n > len(self.df):
            return self.df.head(1) # Return first row if n is larger than dataframe
            
        # Select every nth row starting from a random index
        start_index = np.random.randint(0, n)
        return self.df.iloc[start_index::n]

    def perform_dwt_compression(self, data_series, wavelet='db1', level=1):
        """
        Performs DWT for signal compression.
        """
        # Perform DWT
        coeffs = pywt.wavedec(data_series, wavelet, level=level)
        
        # For compression, we can zero out the detail coefficients, 
        # or in this case, we just reconstruct from the approximation coefficients
        # to show the effect. A more advanced method would threshold the detail coeffs.
        
        # Reconstruct using only approximation coefficients
        # To do this, we create a new coefficient list where all detail coeffs are zero
        compressed_coeffs = [coeffs[0]] + [np.zeros_like(d) for d in coeffs[1:]]
        
        reconstructed_signal = pywt.waverec(compressed_coeffs, wavelet)
        
        # Ensure the reconstructed signal has the same length as the original
        reconstructed_signal = reconstructed_signal[:len(data_series)]
        
        return data_series.values, reconstructed_signal, len(coeffs[0])

    def perform_pca_compression(self, features, n_components):
        """
        Performs PCA for data compression and reconstruction.
        """
        if not all(f in self.df.columns for f in features):
            raise ValueError("指定的特徵中，有一個或多個不存在")
        if n_components > len(features):
            raise ValueError("主成分數量不能超過特徵數量")

        original_data = self.df[features]
        
        # 1. Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(original_data)
        
        # 2. Apply PCA
        pca = PCA(n_components=n_components)
        transformed_data = pca.fit_transform(scaled_data)
        
        # 3. Inverse transform to reconstruct the data
        reconstructed_scaled_data = pca.inverse_transform(transformed_data)
        
        # 4. Inverse scale to get the data back to the original scale
        reconstructed_data = scaler.inverse_transform(reconstructed_scaled_data)
        
        reconstructed_df = pd.DataFrame(reconstructed_data, columns=features, index=original_data.index)
        
        # Calculate reconstruction error (MSE)
        mse = mean_squared_error(original_data, reconstructed_df)
        
        return reconstructed_df, mse, pca

    def get_prediction_data(self, df):
        """Prepare features and target for model training/prediction."""
        features = ['Year_EN', 'Month_NUM', 'Avg_Temperature', 'Science_Park', 'Sub_Science_Park', 'County', 'Town']
        target = 'Electricity_Usage'

        # Filter out rows with NaN in features or target for training
        df_model = df.dropna(subset=features + [target])
        X = df_model[features]
        y = df_model[target]
        
        return X, y, features, target

    def train_predict_evaluate_model(self, model_name, X, y, test_size=0.2, random_state=42, **kwargs):
        """
        Trains, predicts, and evaluates a specified regression model.
        X and y are expected to be the full dataset, before splitting.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # Scale the training and test features
        self.scaler = StandardScaler() # Create a new scaler instance for this training
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        model = None
        if model_name == "LinearRegression":
            model = LinearRegression(**kwargs)
        elif model_name == "DecisionTreeRegressor":
            model = DecisionTreeRegressor(random_state=random_state, **kwargs)
        elif model_name == "HistGradientBoostingRegressor":
            model = HistGradientBoostingRegressor(random_state=random_state, **kwargs)
        elif model_name == "SVR":
            model = SVR(**kwargs)
        elif model_name == "LGBMRegressor":
            model = lgb.LGBMRegressor(random_state=random_state, **kwargs)
        
        if model:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled) # Predict on scaled test set
            
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            metrics = {
                'R2': r2,
                'RMSE': rmse,
                'MAE': mae
            }
            return model, metrics, X_test_scaled, y_test, y_pred
        else:
            raise ValueError(f"Unsupported model: {model_name}")
