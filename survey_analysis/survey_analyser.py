import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
import warnings
import re
from scipy import stats
from scipy.stats import chi2_contingency
from wordcloud import WordCloud, STOPWORDS
from google.colab import files

# Statsmodels for Mosaic Plot
from statsmodels.graphics.mosaicplot import mosaic

# Machine Learning Imports for Multivariate Analysis & Clustering
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sentiment Analysis Import
from textblob import TextBlob

# ==========================================
# 1. GLOBAL CONFIGURATION
# ==========================================

# Updated Categories
CATEGORIES = {
    'Demographic Factors': ['respondents_role', 'childs_age', 'childs_gender', 'age_autism_diagnosed'],
    'Clinical and Behavioural Factors': [
        'childs_autism_severity', 'autism_history_family',
        'comorbidities', 'social_interaction_status', 'emotional_attachment_status',
        'maintain_eye_contact', 'favoured_communication_style', 'unusual_verbal_behaviour',
        'repetitive_behaviours', 'aggressive_behaviour', 'self_injury_incidents',
        'intolerance_to_uncertainty', 'special_education_need',
        'most_challenging_aspect', 'solution_most_challenging_aspect'
    ],
    'Technological Factors': [
        'tech_user_status', 'technology_used', 'purpose_technology_used',
        'child_safety_solution_used', 'belief_in_ai_use',
        'best_information_capture_means', 'device_feedback_expectation'
    ]
}

# Multi-value columns for Word Clouds
MULTI_VALUE_COLS = [
    'comorbidities', 'repetitive_behaviours', 'technology_used',
    'purpose_technology_used', 'child_safety_solution_used',
    'best_information_capture_means', 'device_feedback_expectation',
    'reason_belief', 'most_challenging_aspect', 'solution_most_challenging_aspect'
]

# Columns selected for Sentiment Analysis
OPEN_TEXT_COLS = [
    'most_challenging_aspect',
    'solution_most_challenging_aspect',
    'reason_belief',
    'device_feedback_expectation'
]

# Config for Bivariate/Multivariate Analysis
IV_LIST = ['childs_age', 'childs_gender', 'childs_autism_severity', 'age_autism_diagnosed']
DV_LIST = [
    'social_interaction_status', 'maintain_eye_contact', 'tech_user_status',
    'aggressive_behaviour', 'repetitive_behaviours', 'emotional_attachment_status'
]

# Aliases for Multivariate Class
IV_COLS = IV_LIST
DV_COLS = DV_LIST

SNS_PALETTE = "viridis"
PIE_PALETTE = sns.color_palette('pastel')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def suppress_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    pd.options.mode.chained_assignment = None

def get_file_id(file_path):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    return base_name[-8:] if len(base_name) >= 8 else base_name

def sanitize_text_data(df):
    if df is None: return df
    df = df.copy()
    df_obj = df.select_dtypes(['object'])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())

    if 'repetitive_behaviours' in df.columns:
        def process_smells(text):
            if not isinstance(text, str): return text
            text = text.replace("sounds, textures", "sounds textures")
            match = re.search(r'(smells)(\s*)(.*)', text)
            if match:
                if match.group(3).strip():
                    return text.replace(match.group(0), f"smells, {match.group(3).strip()}")
                else:
                    return text.replace(match.group(0), "smells")
            return text
        df['repetitive_behaviours'] = df['repetitive_behaviours'].apply(process_smells)

    return df.fillna("Not Specified")

def load_dataset(file_path=None):
    if not file_path:
        local_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        if local_files:
             file_path = sorted(local_files)[0]
             print(f"[INFO] Automatically loaded local file: {file_path}")
        else:
            print("Please upload your dataset (CSV or Excel)...")
            uploaded = files.upload()
            if not uploaded:
                print("No file uploaded.")
                return None, None
            file_path = list(uploaded.keys())[0]

    file_id = get_file_id(file_path)

    try:
        df = pd.read_csv(file_path)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='latin1')
        except:
            df = pd.read_excel(file_path)

    df.columns = [c.strip().lower().replace(' ', '_').replace('/', '_').replace('-', '_') for c in df.columns]
    df = sanitize_text_data(df)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    return df, file_id

def get_exploded_series(df, col):
    if col not in df.columns: return pd.Series(dtype='object')
    s = df[col].astype(str).str.split(',').explode().str.strip()
    return s[~s.isin(['nan', 'None', '', 'np.nan', 'Not Specified'])]

def save_plot(fig, output_dir, filename):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)

def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    if min((kcorr-1), (rcorr-1)) == 0: return 0
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

# ==========================================
# 3. ANALYSIS CLASSES
# ==========================================

class SentimentAnalysis:
    def __init__(self, df, file_id, base_dir):
        self.df = df.copy()
        self.file_id = file_id
        self.root_dir = base_dir

        # Folder for Basic Sentiment Analysis
        self.basic_dir = os.path.join(base_dir, f"Sentiment_Analysis_{self.file_id}")

        # Folder for Sentiment Clustering (Requested as separate folder)
        self.cluster_dir = os.path.join(base_dir, f"Sentiment_Clustering_{self.file_id}")

        self._setup_directories()
        suppress_warnings()

    def _setup_directories(self):
        if os.path.exists(self.basic_dir): shutil.rmtree(self.basic_dir)
        os.makedirs(self.basic_dir)

        if os.path.exists(self.cluster_dir): shutil.rmtree(self.cluster_dir)
        os.makedirs(self.cluster_dir)

        # Create subdirectories for each analyzed column in Basic Analysis
        for col in OPEN_TEXT_COLS:
            if col in self.df.columns:
                os.makedirs(os.path.join(self.basic_dir, col), exist_ok=True)
                os.makedirs(os.path.join(self.cluster_dir, col), exist_ok=True)

        os.makedirs(os.path.join(self.basic_dir, "Summary_Data"), exist_ok=True)

    def get_sentiment(self, text):
        if pd.isna(text) or str(text).lower() in ['nan', 'not specified', 'none']:
            return 0, 0, "Neutral"

        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        if polarity > 0.05:
            label = "Positive"
        elif polarity < -0.05:
            label = "Negative"
        else:
            label = "Neutral"

        return polarity, subjectivity, label

    def analyze_columns(self):
        summary_list = []

        for col in OPEN_TEXT_COLS:
            if col not in self.df.columns: continue

            print(f"   Processing Sentiment for: {col}")
            col_dir = os.path.join(self.basic_dir, col)

            # Apply Sentiment Analysis
            self.df[f'{col}_polarity'], self.df[f'{col}_subjectivity'], self.df[f'{col}_sentiment'] = \
                zip(*self.df[col].apply(self.get_sentiment))

            # Filter valid data for plotting
            valid_data = self.df[self.df[col].astype(str).str.lower() != 'not specified']

            if valid_data.empty: continue

            # 1. Sentiment Count Plot
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.countplot(x=f'{col}_sentiment', data=valid_data, order=['Positive', 'Neutral', 'Negative'],
                          palette={'Positive': 'mediumseagreen', 'Neutral': 'lightgray', 'Negative': 'indianred'}, ax=ax)
            ax.set_title(f"Sentiment Distribution: {col}")
            save_plot(fig, col_dir, f"{col}_sentiment_counts.png")

            # 2. Polarity Histogram
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(valid_data[f'{col}_polarity'], kde=True, bins=20, color='teal', ax=ax)
            ax.set_title(f"Polarity Score Distribution: {col}")
            ax.set_xlabel("Polarity (-1 Negative to +1 Positive)")
            save_plot(fig, col_dir, f"{col}_polarity_dist.png")

            # 3. Subjectivity vs Polarity Scatter
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=valid_data, x=f'{col}_polarity', y=f'{col}_subjectivity',
                            hue=f'{col}_sentiment', palette='viridis', alpha=0.6, ax=ax)
            ax.set_title(f"Polarity vs Subjectivity: {col}")
            save_plot(fig, col_dir, f"{col}_polarity_vs_subjectivity.png")

            # 4. Sentiment Word Clouds
            self.generate_sentiment_wordclouds(col, valid_data, col_dir)

            # Record stats
            counts = valid_data[f'{col}_sentiment'].value_counts(normalize=True)
            summary_list.append({
                'Column': col,
                'Positive_Pct': round(counts.get('Positive', 0) * 100, 2),
                'Neutral_Pct': round(counts.get('Neutral', 0) * 100, 2),
                'Negative_Pct': round(counts.get('Negative', 0) * 100, 2),
                'Avg_Polarity': round(valid_data[f'{col}_polarity'].mean(), 3),
                'Avg_Subjectivity': round(valid_data[f'{col}_subjectivity'].mean(), 3)
            })

            # 5. RUN CLUSTERING FOR THIS COLUMN
            self.perform_clustering(col, valid_data)

        # Save Summary Data
        if summary_list:
            pd.DataFrame(summary_list).to_csv(os.path.join(self.basic_dir, "Summary_Data", "Sentiment_Summary.csv"), index=False)

        # Save enriched CSV
        self.df.to_csv(os.path.join(self.basic_dir, "Summary_Data", f"Enriched_Data_with_Sentiment_{self.file_id}.csv"), index=False)

    def generate_sentiment_wordclouds(self, col, df, col_dir):
        for sentiment in ['Positive', 'Negative']:
            subset = df[df[f'{col}_sentiment'] == sentiment]
            text_data = " ".join(subset[col].dropna().astype(str))

            if len(text_data.strip()) < 10: continue

            wc = WordCloud(width=800, height=400, background_color='white',
                           colormap='Greens' if sentiment == 'Positive' else 'Reds',
                           stopwords=STOPWORDS).generate(text_data)

            plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.title(f"{sentiment} Word Cloud: {col}")
            plt.savefig(os.path.join(col_dir, f"WC_{col}_{sentiment}.png"), bbox_inches='tight')
            plt.close()

    def perform_clustering(self, col, df):
        """
        Clusters responses based on Polarity and Subjectivity scores.
        This groups users into categories like 'Passionate Negative', 'Objective Neutral', etc.
        """
        print(f"   -> Clustering sentiment for: {col}")

        # Prepare data for clustering
        X = df[[f'{col}_polarity', f'{col}_subjectivity']].dropna()
        if len(X) < 10:
             print(f"      [SKIP] Not enough data points to cluster {col}")
             return

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # KMeans (k=4 usually captures Quadrants: Pos/Subj, Pos/Obj, Neg/Subj, Neg/Obj)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)

        # Assign clusters back to the subset
        X['Cluster'] = clusters

        # Save cluster ID to main DF for reference
        self.df.loc[X.index, f'{col}_sentiment_cluster'] = clusters

        # Output Directory for this column's clustering
        cluster_out_dir = os.path.join(self.cluster_dir, col)

        # 1. Cluster Visualization (Scatter)
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.scatterplot(x=X[f'{col}_polarity'], y=X[f'{col}_subjectivity'],
                        hue=X['Cluster'], palette='deep', s=100, alpha=0.8, ax=ax)

        # Draw centroids
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        ax.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5, marker='X')

        ax.set_title(f"Sentiment Clustering: {col}\n(Polarity vs Subjectivity)")
        ax.set_xlabel("Polarity")
        ax.set_ylabel("Subjectivity")
        save_plot(fig, cluster_out_dir, f"Cluster_Scatter_{col}.png")

        # 2. Generate Word Cloud per Cluster
        # To understand what people in Cluster X are talking about
        for c_id in sorted(X['Cluster'].unique()):
            # Get indices for this cluster
            idx = X[X['Cluster'] == c_id].index
            # Get text
            text_data = " ".join(df.loc[idx, col].dropna().astype(str))

            if len(text_data.strip()) < 10: continue

            wc = WordCloud(width=800, height=400, background_color='white',
                           colormap='magma', stopwords=STOPWORDS).generate(text_data)

            # Determine avg sentiment for title
            avg_pol = X.loc[idx, f'{col}_polarity'].mean()
            avg_sub = X.loc[idx, f'{col}_subjectivity'].mean()

            plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.title(f"Cluster {c_id} (Pol: {avg_pol:.2f}, Sub: {avg_sub:.2f})")
            plt.savefig(os.path.join(cluster_out_dir, f"WC_Cluster_{c_id}.png"), bbox_inches='tight')
            plt.close()

        # 3. Save Cluster Stats
        stats_df = X.groupby('Cluster').agg({
            f'{col}_polarity': 'mean',
            f'{col}_subjectivity': 'mean',
            f'{col}_polarity': ['mean', 'count']
        })
        stats_df.to_csv(os.path.join(cluster_out_dir, f"Cluster_Stats_{col}.csv"))

    def run(self):
        print("--- Running Sentiment Analysis & Clustering ---")
        self.analyze_columns()


class UnivariateAnalysis:
    def __init__(self, df, file_id, base_dir):
        self.df = df
        self.file_id = file_id
        self.output_dir = os.path.join(base_dir, f"Univariate_Analysis_{self.file_id}")
        self._setup_directories()
        suppress_warnings()

    def _setup_directories(self):
        if os.path.exists(self.output_dir): shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)
        for cat in CATEGORIES.keys():
            os.makedirs(os.path.join(self.output_dir, f"{cat}_{self.file_id}"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, f"Statistical_Results_{self.file_id}"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, f"WordClouds_{self.file_id}"), exist_ok=True)

    def calculate_distribution_stats(self):
        results = []
        all_cols = [c for cat in CATEGORIES.values() for c in cat if c in self.df.columns]
        for col in all_cols:
            is_multi = any(m in col for m in MULTI_VALUE_COLS)
            data = get_exploded_series(self.df, col) if is_multi else self.df[col]
            if data.empty: continue
            codes, uniques = pd.factorize(data)
            counts = data.value_counts()
            if len(counts) > 0:
                max_freq = counts.iloc[0]
                num_modes = sum(counts == max_freq)
                modality = "Unimodal" if num_modes == 1 else "Bimodal" if num_modes == 2 else "Multimodal"
            else: modality = "Undefined"
            results.append({
                'Variable': col, 'Analysis_Type': 'Distribution',
                'Skewness': round(stats.skew(codes), 3), 'Kurtosis': round(stats.kurtosis(codes), 3),
                'Modality': modality, 'Std_Dev': round(np.std(codes), 3), 'IQR': round(stats.iqr(codes), 3)
            })
        return pd.DataFrame(results)

    def conduct_inferential_stats(self):
        results = []
        all_cols = [c for cat in CATEGORIES.values() for c in cat if c in self.df.columns]
        for col in all_cols:
            is_multi = any(m in col for m in MULTI_VALUE_COLS)
            data = get_exploded_series(self.df, col) if is_multi else self.df[col]
            if data.empty: continue
            codes, uniques = pd.factorize(data)
            counts = data.value_counts()
            shapiro_p = stats.shapiro(codes)[1] if len(codes) >= 3 else np.nan
            mid = (len(uniques) - 1) / 2 if len(uniques) > 0 else 0
            ttest_p = stats.ttest_1samp(codes, mid)[1]
            chi_p = stats.chisquare(counts)[1] if len(counts) > 1 else np.nan
            results.append({
                'Variable': col, 'Analysis_Type': 'Inferential',
                'Shapiro_Wilk_P': round(shapiro_p, 4) if pd.notnull(shapiro_p) else np.nan,
                'One_Sample_T_P': round(ttest_p, 4),
                'Chi_Square_P': round(chi_p, 4) if pd.notnull(chi_p) else np.nan
            })
        return pd.DataFrame(results)

    def generate_descriptive_stats(self):
        results = []
        all_cols = [c for cat in CATEGORIES.values() for c in cat if c in self.df.columns]
        for col in all_cols:
            is_multi = any(m in col for m in MULTI_VALUE_COLS)
            data = get_exploded_series(self.df, col) if is_multi else self.df[col]
            total_n = len(data)
            unique_n = data.nunique()
            if not data.empty:
                mode_val = data.mode().iloc[0]
                mode_pct = (data.value_counts().iloc[0] / total_n) * 100
            else: mode_val, mode_pct = "N/A", 0.0
            results.append({
                'Variable': col, 'Analysis_Type': 'Descriptive',
                'Total_N': total_n, 'Unique_Values': unique_n,
                'Mode': mode_val, 'Mode_Percent': round(mode_pct, 2)
            })
        return pd.DataFrame(results)

    def generate_inference(self, row):
        inferences = []
        if pd.notna(row.get('Skewness')):
            skew = row['Skewness']
            direction = "Positive" if skew > 0 else "Negative"
            inferences.append(f"Distribution is {direction.lower()}ly skewed (Skewness={skew}).")
        return " ".join(inferences)

    def generate_visualizations(self):
        sns.set(style="whitegrid")
        for cat_name, cols in CATEGORIES.items():
            valid_cols = [c for c in cols if c in self.df.columns]
            cat_dir = os.path.join(self.output_dir, f"{cat_name}_{self.file_id}")

            for col in valid_cols:
                is_multi = any(m in col for m in MULTI_VALUE_COLS)
                data = get_exploded_series(self.df, col) if is_multi else self.df[col]
                counts = data.value_counts()
                if counts.empty: continue

                col_dir = os.path.join(cat_dir, col)
                os.makedirs(col_dir, exist_ok=True)

                # 1. Bar Chart
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=counts.head(15).values, y=counts.head(15).index, palette=SNS_PALETTE, ax=ax)
                ax.set_title(f"Bar Chart: {col}")
                save_plot(fig, col_dir, f"{col}_bar.png")

                # 2. Pie (Low Cardinality)
                if len(counts) <= 6:
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=PIE_PALETTE)
                    ax.set_title(f"Pie Chart: {col}")
                    save_plot(fig, col_dir, f"{col}_pie.png")

                # 3. Skewness Graph (Histogram with KDE)
                codes, _ = pd.factorize(data)
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(codes, kde=True, bins=min(len(counts), 20), color='teal', ax=ax)
                ax.set_title(f"Distribution (Skewness): {col}")
                ax.set_xlabel("Category Codes")
                save_plot(fig, col_dir, f"{col}_skewness.png")

                # 4. Box Plot
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.boxplot(x=codes, color='lightblue', ax=ax)
                ax.set_title(f"Box Plot: {col}")
                ax.set_xlabel("Category Codes (Distribution)")
                save_plot(fig, col_dir, f"{col}_boxplot.png")

    def generate_word_clouds(self):
        wc_dir = os.path.join(self.output_dir, f"WordClouds_{self.file_id}")
        freq_list = []
        for col in MULTI_VALUE_COLS:
            if col not in self.df.columns: continue
            exploded = get_exploded_series(self.df, col)
            if exploded.empty: continue
            text_data = " ".join(exploded.astype(str))

            wc = WordCloud(width=800, height=400, background_color='white', colormap='viridis', stopwords=STOPWORDS)
            wc_img = wc.generate(text_data)

            plt.figure(figsize=(10, 5))
            plt.imshow(wc_img, interpolation='bilinear')
            plt.axis('off')
            plt.title(f"Word Cloud: {col}")
            plt.savefig(os.path.join(wc_dir, f"WC_{col}.png"))
            plt.close()

            for word, count in wc.process_text(text_data).items():
                freq_list.append({'Variable': col, 'Word': word, 'Frequency': count})

        if freq_list:
            pd.DataFrame(freq_list).to_csv(os.path.join(wc_dir, f"WordCloud_Frequencies_{self.file_id}.csv"), index=False)

    def run(self):
        print("--- Running Univariate Analysis ---")
        df_dist = self.calculate_distribution_stats()
        df_inf = self.conduct_inferential_stats()
        df_desc = self.generate_descriptive_stats()
        combined = pd.concat([df_dist, df_inf, df_desc], axis=0, ignore_index=True)
        if not combined.empty:
            combined['Inference'] = combined.apply(self.generate_inference, axis=1)
            stats_dir = os.path.join(self.output_dir, f"Statistical_Results_{self.file_id}")
            combined.to_csv(os.path.join(stats_dir, f"Combined_Stats_{self.file_id}.csv"), index=False)
        self.generate_visualizations()
        self.generate_word_clouds()


class BivariateAnalysis:
    def __init__(self, df, file_id, base_dir):
        self.df = df
        self.file_id = file_id
        self.output_dir = os.path.join(base_dir, f"Bivariate_Analysis_{self.file_id}")
        self.results = []
        self._setup_directories()
        suppress_warnings()

    def _setup_directories(self):
        if os.path.exists(self.output_dir): shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)
        os.makedirs(os.path.join(self.output_dir, f"Results_{self.file_id}"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, f"Category_Correlations_{self.file_id}"), exist_ok=True)

    def _get_exploded_df(self, iv, dv):
        temp_df = self.df[[iv, dv]].dropna()
        if dv in MULTI_VALUE_COLS:
            temp_df[dv] = temp_df[dv].astype(str).str.split(',')
            temp_df = temp_df.explode(dv)
            temp_df[dv] = temp_df[dv].str.strip()
            temp_df = temp_df[~temp_df[dv].isin(['nan', 'None', '', 'np.nan', 'Not Specified'])]
        return temp_df

    def _get_iv_dir(self, iv):
        return os.path.join(self.output_dir, f"{iv}_{self.file_id}")

    def analyze_pair(self, iv, dv):
        if iv not in self.df.columns or dv not in self.df.columns: return
        temp_df = self._get_exploded_df(iv, dv)
        if temp_df.empty: return
        crosstab = pd.crosstab(temp_df[iv], temp_df[dv])
        if crosstab.shape[0] < 2 or crosstab.shape[1] < 2: return

        chi2, p, dof, expected = chi2_contingency(crosstab)
        n = crosstab.sum().sum()
        min_dim = min(crosstab.shape) - 1
        cramer_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

        inference = f"p={p:.4f}, V={cramer_v:.2f}"
        self.results.append({
            'Independent_Var': iv, 'Dependent_Var': dv,
            'Chi2': round(chi2, 3), 'P_Value': round(p, 5),
            'Cramers_V': round(cramer_v, 3), 'Inference': inference
        })

        save_dir = self._get_iv_dir(iv)
        os.makedirs(save_dir, exist_ok=True)

        crosstab_pct = crosstab.div(crosstab.sum(1), axis=0) * 100
        ax = crosstab_pct.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
        plt.title(f"{iv} vs {dv} (%)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        save_plot(plt.gcf(), save_dir, f"StackedBar_{iv}_vs_{dv}.png")

    def generate_side_by_side(self):
        for dv in DV_LIST:
            if dv not in self.df.columns: continue
            for iv in IV_LIST:
                if iv not in self.df.columns: continue
                exploded_df = self._get_exploded_df(iv, dv)
                if exploded_df.empty: continue

                top_items = exploded_df[dv].value_counts().head(10).index
                plot_data = exploded_df[exploded_df[dv].isin(top_items)]
                if plot_data.empty: continue

                counts = plot_data.groupby([iv, dv]).size().reset_index(name='count')
                group_totals = self.df[iv].value_counts()
                counts['percentage'] = counts.apply(lambda row: (row['count'] / group_totals.get(row[iv], 1)) * 100, axis=1)

                plt.figure(figsize=(12, 7))
                sns.barplot(data=counts, x=dv, y='percentage', hue=iv, palette='viridis', edgecolor='black')
                plt.title(f"Comparison: {dv} by {iv}")
                plt.xticks(rotation=45, ha='right')
                plt.legend(title=iv, bbox_to_anchor=(1.05, 1), loc='upper left')

                save_dir = self._get_iv_dir(iv)
                os.makedirs(save_dir, exist_ok=True)
                save_plot(plt.gcf(), save_dir, f"SideBySide_{iv}_vs_{dv}.png")

    def generate_divergent(self):
        for dv in DV_LIST:
            if dv not in self.df.columns: continue
            for iv in IV_LIST:
                if iv not in self.df.columns: continue

                top_groups = self.df[iv].value_counts().head(2).index.tolist()
                if len(top_groups) < 2: continue
                group_a, group_b = top_groups[0], top_groups[1]
                exploded_df = self._get_exploded_df(iv, dv)
                if exploded_df.empty: continue

                top_items = exploded_df[dv].value_counts().head(10).index
                plot_data = exploded_df[exploded_df[dv].isin(top_items)]
                counts = plot_data.groupby([iv, dv]).size().reset_index(name='count')
                group_totals = self.df[iv].value_counts()
                counts['percentage'] = counts.apply(lambda row: (row['count'] / group_totals.get(row[iv], 1)) * 100, axis=1)

                data_a = counts[counts[iv] == group_a].set_index(dv).reindex(top_items).fillna(0)
                data_b = counts[counts[iv] == group_b].set_index(dv).reindex(top_items).fillna(0)

                fig, ax = plt.subplots(figsize=(12, 8))
                y = np.arange(len(top_items))
                ax.barh(y, -data_a['percentage'], color='skyblue', label=group_a)
                ax.barh(y, data_b['percentage'], color='orange', label=group_b)
                ax.set_yticks(y)
                ax.set_yticklabels(top_items)
                ax.set_title(f"Divergent: {dv} ({group_a} vs {group_b})")
                ax.legend()
                ax.axvline(0, color='black', linewidth=0.8)
                xticks = ax.get_xticks()
                ax.set_xticklabels([f"{abs(x):.0f}%" for x in xticks])

                save_dir = self._get_iv_dir(iv)
                os.makedirs(save_dir, exist_ok=True)
                save_plot(fig, save_dir, f"Divergent_{iv}_vs_{dv}.png")

    def generate_mosaic_plots(self):
        for iv in IV_LIST:
            for dv in DV_LIST:
                if iv not in self.df.columns or dv not in self.df.columns: continue
                temp_df = self._get_exploded_df(iv, dv)
                if temp_df.empty: continue

                top_iv = temp_df[iv].value_counts().head(5).index
                top_dv = temp_df[dv].value_counts().head(5).index
                filtered_df = temp_df[temp_df[iv].isin(top_iv) & temp_df[dv].isin(top_dv)]
                if filtered_df.empty: continue

                fig, ax = plt.subplots(figsize=(12, 8))
                try:
                    mosaic(filtered_df, [iv, dv], ax=ax, title=f"Mosaic: {iv} vs {dv}", gap=0.01, label_rotation=45)
                    plt.tight_layout()
                    save_dir = self._get_iv_dir(iv)
                    os.makedirs(save_dir, exist_ok=True)
                    save_plot(fig, save_dir, f"Mosaic_{iv}_vs_{dv}.png")
                except: plt.close(fig)

    def generate_category_correlation(self):
        save_dir = os.path.join(self.output_dir, f"Category_Correlations_{self.file_id}")
        for cat_name, cols in CATEGORIES.items():
            valid_cols = [c for c in cols if c in self.df.columns]
            if len(valid_cols) < 2: continue
            assoc_matrix = pd.DataFrame(index=valid_cols, columns=valid_cols, dtype=float)
            for col1 in valid_cols:
                for col2 in valid_cols:
                    if col1 == col2:
                        assoc_matrix.loc[col1, col2] = 1.0
                    else:
                        s1 = self.df[col1].astype(str)
                        s2 = self.df[col2].astype(str)
                        conf_matrix = pd.crosstab(s1, s2)
                        assoc_matrix.loc[col1, col2] = cramers_v(conf_matrix)
            plt.figure(figsize=(12, 10))
            sns.heatmap(assoc_matrix.astype(float), annot=True, fmt=".2f", cmap="coolwarm", vmin=0, vmax=1)
            plt.title(f"Variable Correlation (Cramer's V): {cat_name}")
            plt.tight_layout()
            save_plot(plt.gcf(), save_dir, f"Correlation_{cat_name}.png")

    def run(self):
        print("--- Running Bivariate Analysis ---")
        for iv in IV_LIST:
            for dv in DV_LIST:
                self.analyze_pair(iv, dv)

        self.generate_side_by_side()
        self.generate_divergent()
        self.generate_mosaic_plots()
        self.generate_category_correlation()

        if self.results:
            res_df = pd.DataFrame(self.results)
            res_path = os.path.join(self.output_dir, f"Results_{self.file_id}", f"Bivariate_Results_{self.file_id}.csv")
            res_df.to_csv(res_path, index=False)


class MultivariateAnalysis:
    def __init__(self, df, file_id, base_dir):
        self.df = df
        self.file_id = file_id
        self.output_dir = os.path.join(base_dir, f"Multivariate_Analysis_{self.file_id}")
        self.encoded_df = None
        self._setup_directories()
        suppress_warnings()

    def _setup_directories(self):
        if os.path.exists(self.output_dir): shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)
        os.makedirs(os.path.join(self.output_dir, f"Regression_Analysis_{self.file_id}"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, f"Cluster_Analysis_{self.file_id}"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, f"Correlation_Analysis_{self.file_id}"), exist_ok=True)

    def preprocess_data(self):
        self.encoded_df = self.df.copy()
        target_cols = list(set(IV_COLS + DV_COLS)) # Unique columns
        self.encoded_df = self.encoded_df[target_cols].dropna()

        for col in target_cols:
            if col in self.encoded_df.columns:
                if col in MULTI_VALUE_COLS:
                    self.encoded_df[col] = self.encoded_df[col].apply(lambda x: str(x).split(',')[0] if pd.notnull(x) else x)
                le = LabelEncoder()
                self.encoded_df[col] = le.fit_transform(self.encoded_df[col].astype(str))
        print(f"Multivariate Data Shape after preprocessing: {self.encoded_df.shape}")

    def run_correlation(self):
        if self.encoded_df.empty: return
        corr = self.encoded_df.corr(method='spearman')
        save_dir = os.path.join(self.output_dir, f"Correlation_Analysis_{self.file_id}")
        corr.to_csv(os.path.join(save_dir, "Correlation_Matrix.csv"))
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Spearman Correlation")
        save_plot(plt.gcf(), save_dir, "Correlation_Heatmap.png")

    def run_regression(self):
        if self.encoded_df.empty: return
        results = []
        X = self.encoded_df[IV_COLS]
        save_dir = os.path.join(self.output_dir, f"Regression_Analysis_{self.file_id}")

        for dv in DV_COLS:
            if dv not in self.encoded_df.columns: continue
            if self.encoded_df[dv].nunique() < 2: continue
            y = self.encoded_df[dv]
            try:
                model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
                model.fit(X, y)
                acc = accuracy_score(y, model.predict(X))
                impact = np.max(np.abs(model.coef_))
                results.append({'DV': dv, 'Accuracy': acc, 'Max_Coef_Impact': impact})
            except Exception as e: print(f"Regression failed for {dv}: {e}")

        if results:
            res_df = pd.DataFrame(results)
            res_df.to_csv(os.path.join(save_dir, "Regression_Results.csv"), index=False)
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Accuracy', y='DV', data=res_df, palette='viridis')
            plt.title("Regression Model Accuracy per DV")
            save_plot(plt.gcf(), save_dir, "Regression_Accuracy.png")

    def run_cluster(self):
        if self.encoded_df.empty: return
        save_dir = os.path.join(self.output_dir, f"Cluster_Analysis_{self.file_id}")
        X = self.encoded_df[DV_COLS]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        pca = PCA(n_components=2)
        components = pca.fit_transform(X_scaled)
        plt.figure(figsize=(10, 7))
        sns.scatterplot(x=components[:,0], y=components[:,1], hue=clusters, palette='viridis', s=100)
        plt.title("Cluster Analysis (PCA Projection)")
        save_plot(plt.gcf(), save_dir, "Cluster_Visualization.png")

    def run(self):
        print("--- Running Multivariate Analysis ---")
        self.preprocess_data()
        self.run_correlation()
        self.run_regression()
        self.run_cluster()


# ==========================================
# 4. MASTER PIPELINE
# ==========================================

def master_pipeline():
    df, file_id = load_dataset()
    if df is None: return
    root_dir = f"survey_analysis_{file_id}"
    if os.path.exists(root_dir): shutil.rmtree(root_dir)
    os.makedirs(root_dir)
    print(f"\n[INFO] Root Output Directory: {root_dir}")
    try:
        # 1. Univariate
        ua = UnivariateAnalysis(df, file_id, root_dir)
        ua.run()

        # 2. Sentiment Analysis (Analysis + Clustering)
        sa = SentimentAnalysis(df, file_id, root_dir)
        sa.run()

        # 3. Bivariate
        ba = BivariateAnalysis(df, file_id, root_dir)
        ba.run()

        # 4. Multivariate
        ma = MultivariateAnalysis(df, file_id, root_dir)
        ma.run()

    except Exception as e:
        print(f"[ERROR] Analysis pipeline failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Archiving Results ---")
    shutil.make_archive(root_dir, 'zip', root_dir)
    print(f"[SUCCESS] Zip file created: {root_dir}.zip")

    try:
        files.download(f"{root_dir}.zip")
    except:
        print("Download not available in this environment.")

if __name__ == "__main__":
    master_pipeline()
