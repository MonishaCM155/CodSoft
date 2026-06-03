# ================================================================
# TASK 2: MOVIE RATING PREDICTION WITH PYTHON
# Dataset: IMDb Movies India
# ================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

print("=" * 65)
print("   TASK 2: MOVIE RATING PREDICTION — IMDb India Dataset")
print("=" * 65)

# ── 1. LOAD DATA ──────────────────────────────────────────────────
df = pd.read_csv('/home/claude/datasets/IMDb Movies India.csv', encoding='latin1')
print(f"\n📦 Raw dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── 2. DATA CLEANING ──────────────────────────────────────────────
print("\n🔧 Cleaning data...")

# Year: "(2019)" → 2019
df['Year'] = df['Year'].str.extract(r'(\d{4})').astype(float)

# Duration: "109 min" → 109
df['Duration'] = df['Duration'].str.extract(r'(\d+)').astype(float)

# Votes: remove commas, handle "$5.16M" style outliers → numeric
def parse_votes(v):
    if pd.isna(v):
        return np.nan
    v = str(v).replace(',', '').strip()
    if v.startswith('$') or not v.replace('.','').replace('-','').isdigit():
        return np.nan
    return float(v)

df['Votes'] = df['Votes'].apply(parse_votes)

# Drop rows with no Rating (target)
df = df.dropna(subset=['Rating'])
print(f"   Rows after dropping null Rating: {len(df):,}")

# ── 3. FEATURE ENGINEERING ────────────────────────────────────────
print("\n⚙️  Feature engineering...")

# Primary genre (first listed)
df['Primary_Genre'] = df['Genre'].str.split(',').str[0].str.strip().fillna('Unknown')
df['Num_Genres']    = df['Genre'].str.split(',').str.len().fillna(1)

# Movie era / age
df['Year']      = df['Year'].fillna(df['Year'].median())
df['Movie_Age'] = 2024 - df['Year']

# Director & Actor reputation: their average rating across all movies
for col in ['Director', 'Actor 1', 'Actor 2', 'Actor 3']:
    mean_map = df.groupby(col)['Rating'].mean()
    df[f'{col}_avg'] = df[col].map(mean_map)

# Log votes (popularity)
df['Log_Votes'] = np.log1p(df['Votes'].fillna(0))

# Duration: fill with median
df['Duration'] = df['Duration'].fillna(df['Duration'].median())

# Encode primary genre
le_genre = LabelEncoder()
df['Genre_Encoded'] = le_genre.fit_transform(df['Primary_Genre'])

print(f"   Feature engineering done. Rows: {len(df):,}")

# ── 4. EDA ────────────────────────────────────────────────────────
print("\n📊 Generating EDA plots...")

PURPLE = '#6C5CE7'
GREEN  = '#00B894'
ORANGE = '#FDCB6E'
RED    = '#E17055'

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle('IMDb India Movies — Exploratory Data Analysis',
             fontsize=18, fontweight='bold', y=1.01)

# Rating distribution
axes[0,0].hist(df['Rating'], bins=40, color=PURPLE, edgecolor='white', alpha=0.85)
axes[0,0].axvline(df['Rating'].mean(), color='red', linestyle='--', linewidth=1.5,
                   label=f"Mean = {df['Rating'].mean():.2f}")
axes[0,0].set_title('Rating Distribution'); axes[0,0].set_xlabel('Rating')
axes[0,0].set_ylabel('Count'); axes[0,0].legend()

# Avg rating by top genres
top_genres = df['Primary_Genre'].value_counts().head(8).index
genre_means = df[df['Primary_Genre'].isin(top_genres)].groupby('Primary_Genre')['Rating'].mean().sort_values()
genre_means.plot(kind='barh', ax=axes[0,1], color=PURPLE, alpha=0.85)
axes[0,1].set_title('Avg Rating by Genre (Top 8)'); axes[0,1].set_xlabel('Avg IMDb Rating')

# Log Votes vs Rating
axes[0,2].scatter(df['Log_Votes'], df['Rating'], alpha=0.2, s=8, color=PURPLE)
axes[0,2].set_title('Popularity vs Rating')
axes[0,2].set_xlabel('Log(Votes)'); axes[0,2].set_ylabel('Rating')

# Duration vs Rating
mask_d = df['Duration'] < 300
axes[1,0].scatter(df.loc[mask_d,'Duration'], df.loc[mask_d,'Rating'], alpha=0.2, s=8, color=GREEN)
axes[1,0].set_title('Duration vs Rating')
axes[1,0].set_xlabel('Duration (min)'); axes[1,0].set_ylabel('Rating')

# Avg rating over years
yearly = df.groupby('Year')['Rating'].mean().loc[lambda s: s.index >= 1960]
axes[1,1].plot(yearly.index, yearly.values, color=PURPLE, linewidth=2)
axes[1,1].fill_between(yearly.index, yearly.values, alpha=0.15, color=PURPLE)
axes[1,1].set_title('Avg Rating Over Years'); axes[1,1].set_xlabel('Year')

# Top directors (≥5 movies)
dir_counts = df['Director'].value_counts()
qualified  = dir_counts[dir_counts >= 5].index
dir_avg    = df[df['Director'].isin(qualified)].groupby('Director')['Rating'].mean()
top10_dir  = dir_avg.sort_values(ascending=False).head(10).sort_values()
top10_dir.plot(kind='barh', ax=axes[1,2], color=ORANGE, alpha=0.9)
axes[1,2].set_title('Top 10 Directors by Avg Rating (≥5 movies)')
axes[1,2].set_xlabel('Avg Rating')

# Genre count vs rating
ng = df.groupby('Num_Genres')['Rating'].mean().sort_index()
ng.plot(kind='bar', ax=axes[2,0], color=RED, alpha=0.85)
axes[2,0].set_title('Avg Rating by Number of Genres')
axes[2,0].set_xticklabels(axes[2,0].get_xticklabels(), rotation=0)

# Correlation heatmap
corr_cols = ['Rating','Duration','Year','Log_Votes','Num_Genres',
             'Director_avg','Actor 1_avg','Actor 2_avg']
corr_data = df[corr_cols].dropna()
corr = corr_data.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            ax=axes[2,1], linewidths=0.4, annot_kws={'size': 8})
axes[2,1].set_title('Feature Correlation Heatmap')

# Rating by era
df['Era'] = pd.cut(df['Year'],
                   bins=[1940,1980,1990,2000,2010,2015,2025],
                   labels=['Pre-80s','80s','90s','2000s','2010–15','2015+'])
era_mean = df.groupby('Era', observed=True)['Rating'].mean()
era_count = df.groupby('Era', observed=True)['Rating'].count()
era_mean.plot(kind='bar', ax=axes[2,2], color=PURPLE, alpha=0.85)
axes[2,2].set_title('Avg Rating by Era')
axes[2,2].set_xticklabels(axes[2,2].get_xticklabels(), rotation=30)
for i, (v, c) in enumerate(zip(era_mean, era_count)):
    axes[2,2].text(i, v + 0.02, f'n={c}', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('/home/claude/CODSOFT/task2_eda.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ EDA plots saved")

# ── 5. PREPARE FEATURE MATRIX ─────────────────────────────────────
print("\n🏗️  Preparing feature matrix...")

feature_cols = [
    'Duration', 'Year', 'Movie_Age', 'Log_Votes',
    'Num_Genres', 'Genre_Encoded',
    'Director_avg', 'Actor 1_avg', 'Actor 2_avg', 'Actor 3_avg'
]

df_model = df[feature_cols + ['Rating']].copy()
# Only keep rows where Director avg is available (key feature)
df_model = df_model.dropna(subset=['Director_avg'])

imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(df_model[feature_cols]), columns=feature_cols)
y = df_model['Rating'].values

print(f"   Samples: {X.shape[0]:,}  |  Features: {X.shape[1]}")
print(f"   Feature list: {feature_cols}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

# ── 6. TRAIN & EVALUATE MODELS ────────────────────────────────────
print("\n🤖 Training models...")

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression':  Ridge(alpha=1.0),
    'Random Forest':     RandomForestRegressor(n_estimators=200, max_depth=12,
                                               min_samples_leaf=3, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                                   learning_rate=0.05, random_state=42)
}

results = {}
print(f"\n{'Model':<22} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
print("-" * 52)
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = np.clip(model.predict(X_test), 1.0, 10.0)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)
    results[name] = {'model': model, 'preds': preds, 'rmse': rmse, 'mae': mae, 'r2': r2}
    print(f"{name:<22} {rmse:>8.4f} {mae:>8.4f} {r2:>8.4f}")

best_name = max(results, key=lambda k: results[k]['r2'])
best = results[best_name]
print(f"\n🏆 Best Model: {best_name}  (R²={best['r2']:.4f})")

cv_scores = cross_val_score(models[best_name], X, y, cv=5, scoring='r2')
print(f"   5-Fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── 7. RESULTS VISUALIZATION ──────────────────────────────────────
print("\n📊 Generating results plots...")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Movie Rating Prediction — Model Evaluation', fontsize=16, fontweight='bold')

model_names = list(results.keys())
r2_vals   = [results[n]['r2']   for n in model_names]
rmse_vals = [results[n]['rmse'] for n in model_names]
bar_colors = [RED if n == best_name else '#b2bec3' for n in model_names]

# R² comparison
bars = axes[0,0].bar(model_names, r2_vals, color=bar_colors, edgecolor='white', linewidth=1.2)
axes[0,0].set_title('R² Score Comparison'); axes[0,0].set_ylabel('R² Score')
axes[0,0].set_ylim(0, 1); axes[0,0].tick_params(axis='x', rotation=18)
for bar, v in zip(bars, r2_vals):
    axes[0,0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                   f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')

# RMSE comparison
bars2 = axes[0,1].bar(model_names, rmse_vals, color=bar_colors, edgecolor='white', linewidth=1.2)
axes[0,1].set_title('RMSE Comparison (lower = better)'); axes[0,1].set_ylabel('RMSE')
axes[0,1].tick_params(axis='x', rotation=18)
for bar, v in zip(bars2, rmse_vals):
    axes[0,1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                   f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')

# Actual vs Predicted
axes[0,2].scatter(y_test, best['preds'], alpha=0.25, s=10, color=PURPLE)
mn, mx = min(y_test.min(), best['preds'].min()), max(y_test.max(), best['preds'].max())
axes[0,2].plot([mn,mx],[mn,mx],'r--', linewidth=1.5, label='Perfect fit')
axes[0,2].set_xlabel('Actual Rating'); axes[0,2].set_ylabel('Predicted Rating')
axes[0,2].set_title(f'Actual vs Predicted — {best_name}')
axes[0,2].legend()
axes[0,2].text(0.05, 0.90, f"R²  = {best['r2']:.4f}\nRMSE= {best['rmse']:.4f}",
               transform=axes[0,2].transAxes, fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Residuals vs Predicted
residuals = y_test - best['preds']
axes[1,0].scatter(best['preds'], residuals, alpha=0.25, s=10, color=GREEN)
axes[1,0].axhline(0, color='red', linestyle='--', linewidth=1.5)
axes[1,0].set_xlabel('Predicted Rating'); axes[1,0].set_ylabel('Residual')
axes[1,0].set_title('Residual Plot')

# Residual histogram
axes[1,1].hist(residuals, bins=40, color=ORANGE, edgecolor='white', alpha=0.9)
axes[1,1].axvline(0, color='red', linestyle='--', linewidth=1.5)
axes[1,1].set_title(f'Residual Distribution  (mean={residuals.mean():.3f})')
axes[1,1].set_xlabel('Residual'); axes[1,1].set_ylabel('Count')

# Feature importance
best_tree = results['Random Forest']['model'] if 'Random Forest' in results else results[best_name]['model']
fi = pd.Series(best_tree.feature_importances_, index=feature_cols).sort_values()
fi_colors = [RED if v >= fi.quantile(0.75) else PURPLE if v >= fi.median() else '#b2bec3' for v in fi]
fi.plot(kind='barh', ax=axes[1,2], color=fi_colors)
axes[1,2].set_title('Feature Importance (Random Forest)')
axes[1,2].set_xlabel('Importance Score')

plt.tight_layout()
plt.savefig('/home/claude/CODSOFT/task2_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Results plots saved")

# ── 8. FINAL SUMMARY ──────────────────────────────────────────────
print(f"""
{'='*65}
✅  TASK 2 COMPLETE — FINAL SUMMARY
{'='*65}
  Dataset     : IMDb Movies India  ({len(df):,} movies, post-clean)
  Target      : IMDb Rating (1.0 – 10.0)
  Features    : {len(feature_cols)} engineered features

  ┌──────────────────────────┬────────┬────────┬────────┐
  │ Model                    │  RMSE  │  MAE   │   R²   │
  ├──────────────────────────┼────────┼────────┼────────┤""")
for name in model_names:
    r = results[name]
    star = " ⭐" if name == best_name else "   "
    print(f"  │ {name:<24s} │ {r['rmse']:6.4f} │ {r['mae']:6.4f} │ {r['r2']:6.4f} │{star}")
print(f"""  └──────────────────────────┴────────┴────────┴────────┘

  🏆 Best Model  : {best_name}
  R² Score       : {best['r2']:.4f}  → explains {best['r2']*100:.1f}% of rating variance
  RMSE           : {best['rmse']:.4f}  → avg error ≈ {best['rmse']:.2f} rating points
  5-Fold CV R²   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}
{'='*65}
""")
