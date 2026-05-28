# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:29:42 2026

@author: Lenovo
"""
"""
Mental Health in Technology-related Jobs
Case Study: Employee Segmentation via K-Means Clustering
Course: DLBDSMLUSL01 – Machine Learning – Unsupervised Learning and Feature Engineering

Dataset: OSMI Mental Health in Tech Survey 2016 (Real dataset, 1433 respondents, 63 questions)
Source:  https://www.kaggle.com/datasets/osmi/mental-health-in-tech-2016
         Also available at:
         https://raw.githubusercontent.com/techtenant/OSMI-Mental-Health-in-Tech-Survey/
         master/mental-heath-in-tech-2016_20161114.csv

HOW TO RUN:
1. Download the CSV from Kaggle (link above) and save it as:
       mental-heath-in-tech-2016_20161114.csv
   in the SAME folder as this script.
2. Run:  python mental_health_clustering.py
3. All 6 figures will be saved to a 'figures/' folder automatically.
"""

# ─────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

os.makedirs('figures', exist_ok=True)

# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────
print('=' * 60)
print('STEP 1: DATA LOADING')
print('=' * 60)

CSV_FILE = 'mental-heath-in-tech-2016_20161114.csv'

# If the file isn't found locally, download it automatically
if not os.path.exists(CSV_FILE):
    print(f'  {CSV_FILE} not found locally – downloading...')
    import urllib.request
    url = ('https://raw.githubusercontent.com/techtenant/'
           'OSMI-Mental-Health-in-Tech-Survey/master/'
           'mental-heath-in-tech-2016_20161114.csv')
    urllib.request.urlretrieve(url, CSV_FILE)
    print('  Download complete.')

df_raw = pd.read_csv(CSV_FILE)
print(f'  Loaded: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns')

# ─────────────────────────────────────────────
# 2. FEATURE SELECTION & RENAMING
# ─────────────────────────────────────────────
print('\n' + '=' * 60)
print('STEP 2: FEATURE SELECTION')
print('=' * 60)

# Select 15 most relevant columns and give them short readable names
col_map = {
    'What is your age?':
        'Age',
    'What is your gender?':
        'Gender',
    'Are you self-employed?':
        'self_employed',
    'Is your employer primarily a tech company/organization?':
        'tech_company',
    'Do you work remotely?':
        'remote_work',
    'Does your employer provide mental health benefits as part of healthcare coverage?':
        'mh_benefits',
    'Do you know the options for mental health care available under your employer-provided coverage?':
        'care_options',
    'Does your employer offer resources to learn more about mental health concerns and options for seeking help?':
        'seek_help',
    'Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?':
        'anonymity',
    'If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:':
        'leave_ease',
    'Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?':
        'supervisor_comfort',
    'Would you feel comfortable discussing a mental health disorder with your coworkers?':
        'coworker_comfort',
    'Do you feel that your employer takes mental health as seriously as physical health?':
        'mh_vs_ph_employer',
    'Have you had a mental health disorder in the past?':
        'mh_history_past',
    'Do you currently have a mental health disorder?':
        'mh_current',
    'Have you ever sought treatment for a mental health issue from a mental health professional?':
        'treatment_sought',
    'If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?':
        'work_interfere_untreated',
    'Do you have a family history of mental illness?':
        'family_mh_history',
}

df = df_raw[list(col_map.keys())].rename(columns=col_map).copy()
print(f'  Selected {df.shape[1]} features')
print(f'  Shape: {df.shape}')

# ─────────────────────────────────────────────
# 3. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
print('\n' + '=' * 60)
print('STEP 3: EXPLORATORY DATA ANALYSIS')
print('=' * 60)

print('\n  Age statistics:')
print(df['Age'].describe().round(1).to_string())

print('\n  Mental health (past) distribution:')
print(df['mh_history_past'].value_counts().to_string())

print('\n  Currently has MH disorder:')
print(df['mh_current'].value_counts().to_string())

print('\n  Treatment sought:')
print(df['treatment_sought'].value_counts().to_string())

# Figure 1: EDA
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Figure 1: Exploratory Data Analysis – Key Survey Features',
             fontsize=13, fontweight='bold')

# MH history bar
mh_counts = df['mh_history_past'].value_counts()
colors = ['#E74C3C', '#2ECC71', '#F39C12']
axes[0].bar(mh_counts.index, mh_counts.values,
            color=colors[:len(mh_counts)], edgecolor='white')
axes[0].set_title('Past Mental Health Disorder', fontsize=11)
axes[0].set_xlabel('Response')
axes[0].set_ylabel('Count')
for i, v in enumerate(mh_counts.values):
    axes[0].text(i, v + 8, str(v), ha='center', fontweight='bold', fontsize=9)

# Work interference bar
wi_counts = df['work_interfere_untreated'].value_counts()
colors2 = ['#C0392B', '#E67E22', '#F1C40F', '#27AE60', '#95A5A6', '#3498DB']
axes[1].bar(range(len(wi_counts)), wi_counts.values,
            color=colors2[:len(wi_counts)], edgecolor='white')
axes[1].set_xticks(range(len(wi_counts)))
axes[1].set_xticklabels(wi_counts.index, rotation=20, ha='right', fontsize=9)
axes[1].set_title('Work Interference When NOT Treated', fontsize=11)
axes[1].set_xlabel('Level of Interference')
axes[1].set_ylabel('Count')
for i, v in enumerate(wi_counts.values):
    axes[1].text(i, v + 5, str(v), ha='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('figures/figure1_eda.png', dpi=150, bbox_inches='tight')
plt.close()
print('\n  → Saved figures/figure1_eda.png')

# ─────────────────────────────────────────────
# 4. DATA PREPROCESSING
# ─────────────────────────────────────────────
print('\n' + '=' * 60)
print('STEP 4: DATA PREPROCESSING')
print('=' * 60)

df_clean = df.copy()

# 4.1 Age: remove outliers
before = len(df_clean)
df_clean = df_clean[(df_clean['Age'] >= 18) & (df_clean['Age'] <= 70)]
print(f'  Age filter: removed {before - len(df_clean)} outliers → {len(df_clean)} rows remain')

# 4.2 Gender standardisation
def std_gender(g):
    g = str(g).strip().lower()
    if any(x in g for x in ['male', ' m', '^m$']):
        if 'female' not in g and 'woman' not in g:
            return 'Male'
    if any(x in g for x in ['female', 'woman', 'girl', 'f']):
        return 'Female'
    return 'Other'

df_clean['Gender'] = df_clean['Gender'].apply(std_gender)
print(f'  Gender distribution:\n{df_clean["Gender"].value_counts().to_string()}')

# 4.3 Missing value report
missing = df_clean.isnull().sum()
print(f'\n  Missing values:\n{missing[missing > 0].to_string() if missing.sum() > 0 else "  None"}')

# Drop rows with any remaining NaN
before2 = len(df_clean)
df_clean = df_clean.dropna()
print(f'  After dropping NaN rows: {before2 - len(df_clean)} removed → {len(df_clean)} remain')

# 4.4 Features for clustering (drop Gender – too many 'Other'; drop Age for now – keep numeric)
features = [
    'self_employed', 'tech_company', 'remote_work',
    'mh_benefits', 'care_options', 'seek_help', 'anonymity', 'leave_ease',
    'supervisor_comfort', 'coworker_comfort', 'mh_vs_ph_employer',
    'mh_history_past', 'mh_current', 'treatment_sought',
    'work_interfere_untreated', 'family_mh_history',
]

df_model = df_clean[features].copy()

# 4.5 Label encoding
le = LabelEncoder()
for col in df_model.columns:
    df_model[col] = le.fit_transform(df_model[col].astype(str))

df_model = df_model.apply(pd.to_numeric)
print(f'\n  Model feature matrix shape: {df_model.shape}')

# 4.6 Standard scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_model)

# Figure 2: Preprocessing overview
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Figure 2: Data Preprocessing Overview', fontsize=13, fontweight='bold')

missing_pct = (df.isnull().sum() / len(df) * 100)
axes[0].barh(missing_pct.index, missing_pct.values, color='#3498DB')
axes[0].set_title('Missing Values per Feature (%)', fontsize=11)
axes[0].set_xlabel('% Missing')
axes[0].axvline(0, color='grey', linewidth=0.8)

corr = pd.DataFrame(X_scaled[:, :8], columns=features[:8]).corr()
sns.heatmap(corr, ax=axes[1], cmap='RdBu_r', center=0, annot=False,
            linewidths=0.3, cbar_kws={'shrink': 0.7})
axes[1].set_title('Feature Correlation (first 8 features)', fontsize=11)
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('figures/figure2_preprocessing.png', dpi=150, bbox_inches='tight')
plt.close()
print('  → Saved figures/figure2_preprocessing.png')

# ─────────────────────────────────────────────
# 5. FEATURE ENGINEERING: PCA
# ─────────────────────────────────────────────
print('\n' + '=' * 60)
print('STEP 5: DIMENSIONALITY REDUCTION (PCA)')
print('=' * 60)

pca_full = PCA().fit(X_scaled)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_components = int(np.argmax(cumvar >= 0.80)) + 1
print(f'  Components for ≥80% variance: {n_components}')
print(f'  Actual variance explained: {cumvar[n_components-1]*100:.1f}%')

pca = PCA(n_components=n_components, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f'  PCA output shape: {X_pca.shape}')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Figure 3: PCA – Explained Variance Analysis', fontsize=13, fontweight='bold')

axes[0].bar(range(1, len(pca_full.explained_variance_ratio_) + 1),
            pca_full.explained_variance_ratio_ * 100, color='#2E86AB', edgecolor='white')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Explained Variance (%)')
axes[0].set_title('Individual Explained Variance')
axes[0].set_xlim(0.5, len(features) + 0.5)

axes[1].plot(range(1, len(cumvar) + 1), cumvar * 100, 'o-', color='#E74C3C', linewidth=2)
axes[1].axhline(80, color='grey', linestyle='--', label='80% threshold')
axes[1].axvline(n_components, color='#27AE60', linestyle='--',
                label=f'n={n_components} components')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Explained Variance (%)')
axes[1].set_title('Cumulative Explained Variance')
axes[1].legend()

plt.tight_layout()
plt.savefig('figures/figure3_pca.png', dpi=150, bbox_inches='tight')
plt.close()
print('  → Saved figures/figure3_pca.png')

# ─────────────────────────────────────────────
# 6. K-MEANS CLUSTERING
# ─────────────────────────────────────────────
print('\n' + '=' * 60)
print('STEP 6: K-MEANS CLUSTERING')
print('=' * 60)

sse_values, sil_values = [], []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    km.fit(X_pca)
    sse_values.append(km.inertia_)
    sil = silhouette_score(X_pca, km.labels_, sample_size=500, random_state=42)
    sil_values.append(sil)
    print(f'  k={k}: SSE={km.inertia_:.1f}, Silhouette={sil:.4f}')

# Select k=4 (good balance of elbow + interpretability)
final_k = 4
print(f'\n  Selected k={final_k}')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Figure 4: Optimal Cluster Determination', fontsize=13, fontweight='bold')

axes[0].plot(list(k_range), sse_values, 'o-', color='#2E86AB', linewidth=2, markersize=6)
axes[0].axvline(final_k, color='#E74C3C', linestyle='--', label=f'Selected k={final_k}')
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Sum of Squared Errors (SSE)')
axes[0].set_title('Elbow Method')
axes[0].legend()

axes[1].plot(list(k_range), sil_values, 's-', color='#27AE60', linewidth=2, markersize=6)
axes[1].axvline(final_k, color='#E74C3C', linestyle='--', label=f'Selected k={final_k}')
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Analysis')
axes[1].legend()

plt.tight_layout()
plt.savefig('figures/figure4_elbow_silhouette.png', dpi=150, bbox_inches='tight')
plt.close()
print('  → Saved figures/figure4_elbow_silhouette.png')

# Final model
kmeans = KMeans(n_clusters=final_k, init='k-means++', n_init=10, random_state=42)
kmeans.fit(X_pca)
df_clean = df_clean.iloc[:len(kmeans.labels_)].copy()
df_clean['Cluster'] = kmeans.labels_

final_sil = silhouette_score(X_pca, kmeans.labels_, sample_size=500, random_state=42)
print(f'\n  Final silhouette score: {final_sil:.4f}')

# ─────────────────────────────────────────────
# 7. RESULTS & VISUALISATION
# ─────────────────────────────────────────────
print('\n' + '=' * 60)
print('STEP 7: RESULTS & CLUSTER ANALYSIS')
print('=' * 60)

pca_2d = PCA(n_components=2, random_state=42)
X_2d = pca_2d.fit_transform(X_scaled[:len(kmeans.labels_)])

cluster_counts = df_clean['Cluster'].value_counts().sort_index()
print(f'\n  Cluster distribution:\n{cluster_counts.to_string()}')

palette = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Figure 5: Employee Cluster Visualisation', fontsize=13, fontweight='bold')

for c in range(final_k):
    mask = df_clean['Cluster'].values == c
    axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1],
                    c=palette[c], label=f'Cluster {c}',
                    alpha=0.5, s=10, edgecolors='none')

axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
axes[0].set_title('PCA 2D Projection of Clusters')
axes[0].legend(markerscale=3)

axes[1].bar(cluster_counts.index, cluster_counts.values,
            color=palette[:final_k], edgecolor='white')
axes[1].set_xlabel('Cluster')
axes[1].set_ylabel('Number of Respondents')
axes[1].set_title('Cluster Distribution')
for i, v in enumerate(cluster_counts.values):
    pct = v / cluster_counts.sum() * 100
    axes[1].text(i, v + 3, f'{v}\n({pct:.1f}%)', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/figure5_cluster_viz.png', dpi=150, bbox_inches='tight')
plt.close()
print('  → Saved figures/figure5_cluster_viz.png')

# Cluster profile heatmap
df_model_clustered = df_model.copy()
df_model_clustered['Cluster'] = kmeans.labels_
profile_cols = ['treatment_sought', 'work_interfere_untreated', 'mh_history_past',
                'mh_current', 'mh_benefits', 'supervisor_comfort',
                'coworker_comfort', 'mh_vs_ph_employer', 'family_mh_history']
cluster_profiles = df_model_clustered.groupby('Cluster')[profile_cols].mean().round(2)

fig, ax = plt.subplots(figsize=(13, 5))
sns.heatmap(cluster_profiles, annot=True, fmt='.2f', cmap='YlOrRd',
            linewidths=0.5, ax=ax, cbar_kws={'label': 'Mean encoded value'})
ax.set_title('Figure 6: Cluster Feature Profile Heatmap', fontsize=13, fontweight='bold')
ax.set_xlabel('Feature')
ax.set_ylabel('Cluster')
plt.tight_layout()
plt.savefig('figures/figure6_cluster_profile.png', dpi=150, bbox_inches='tight')
plt.close()
print('  → Saved figures/figure6_cluster_profile.png')

# ─────────────────────────────────────────────
# 8. CLUSTER INTERPRETATION
# ─────────────────────────────────────────────
print('\n' + '=' * 60)
print('STEP 8: CLUSTER INTERPRETATION')
print('=' * 60)

for c in range(final_k):
    subset = df_clean[df_clean['Cluster'] == c]
    print(f'\n  --- Cluster {c} (n={len(subset)}) ---')
    print(f'    MH history (past):      {subset["mh_history_past"].value_counts().index[0]}')
    print(f'    Currently has MH:       {subset["mh_current"].value_counts().index[0]}')
    print(f'    Treatment sought:       {subset["treatment_sought"].value_counts().index[0]}')
    print(f'    MH benefits provided:   {subset["mh_benefits"].value_counts().index[0]}')
    print(f'    Supervisor comfort:     {subset["supervisor_comfort"].value_counts().index[0]}')
    print(f'    Work interferes(untx):  {subset["work_interfere_untreated"].value_counts().index[0]}')

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print('\n' + '=' * 60)
print('SUMMARY')
print('=' * 60)
print(f'  Dataset:          OSMI 2016 (real) – {len(df_clean)} respondents after cleaning')
print(f'  Features used:    {len(features)}')
print(f'  PCA components:   {n_components} (explaining {cumvar[n_components-1]*100:.1f}% variance)')
print(f'  Final k:          {final_k}')
print(f'  Silhouette score: {final_sil:.4f}')
print(f'\n  All 6 figures saved to ./figures/')
print('\nAnalysis complete.')
