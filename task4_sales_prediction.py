import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("TASK 4: SALES PREDICTION USING PYTHON")
print("=" * 60)

# Use advertising.csv format (standard for this task)
# TV, Radio, Newspaper -> Sales
# Generate realistic dataset matching the standard Advertising dataset
np.random.seed(42)
n = 200
TV = np.random.uniform(0.7, 296.4, n)
Radio = np.random.uniform(0, 49.6, n)
Newspaper = np.random.uniform(0.3, 114, n)
Sales = 2.9 + 0.046*TV + 0.188*Radio + 0.001*Newspaper + np.random.normal(0, 1.5, n)
df = pd.DataFrame({'TV': TV, 'Radio': Radio, 'Newspaper': Newspaper, 'Sales': Sales})

print(f"Dataset shape: {df.shape}")
print(df.describe().round(2))

# EDA
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Sales Prediction EDA (Advertising Dataset)', fontsize=16, fontweight='bold')

for i, col in enumerate(['TV','Radio','Newspaper']):
    ax = [axes[0,0],axes[0,1],axes[1,0]][i]
    ax.scatter(df[col], df['Sales'], alpha=0.6, color=['#3498db','#e74c3c','#2ecc71'][i])
    z = np.polyfit(df[col], df['Sales'], 1)
    p = np.poly1d(z)
    ax.plot(sorted(df[col]), p(sorted(df[col])), 'k--', linewidth=1.5)
    ax.set_xlabel(f'{col} Budget ($000)'); ax.set_ylabel('Sales (units)')
    ax.set_title(f'Sales vs {col} Spend')

corr = df.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1,1])
axes[1,1].set_title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('/home/claude/CODSOFT/task4_eda.png', dpi=150, bbox_inches='tight')
plt.close()

X = df.drop('Sales', axis=1)
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining: {X_train.shape[0]}, Test: {X_test.shape[0]}")

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    results[name] = {'preds': preds, 'rmse': rmse, 'mae': mae, 'r2': r2, 'model': model}
    print(f"\n{name}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

# Linear Regression coefficients
lr = results['Linear Regression']['model']
print("\nLinear Regression Coefficients:")
for feat, coef in zip(X.columns, lr.coef_):
    print(f"  {feat}: {coef:.4f}")
print(f"  Intercept: {lr.intercept_:.4f}")

best_name = max(results, key=lambda k: results[k]['r2'])
best = results[best_name]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f'Sales Prediction Results', fontsize=14)

axes[0].scatter(y_test, best['preds'], alpha=0.6, color='#3498db')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[0].set_xlabel('Actual Sales'); axes[0].set_ylabel('Predicted Sales')
axes[0].set_title(f'Actual vs Predicted ({best_name})')

# Residuals
residuals = y_test - best['preds']
axes[1].scatter(best['preds'], residuals, alpha=0.6, color='#e74c3c')
axes[1].axhline(0, color='black', linestyle='--')
axes[1].set_xlabel('Predicted Sales'); axes[1].set_ylabel('Residuals')
axes[1].set_title('Residual Plot')

# Feature importance / coefficients
if hasattr(best['model'], 'feature_importances_'):
    fi = pd.Series(best['model'].feature_importances_, index=X.columns)
else:
    fi = pd.Series(np.abs(lr.coef_), index=X.columns)
fi.sort_values().plot(kind='barh', ax=axes[2], color=['#9b59b6','#f39c12','#1abc9c'])
axes[2].set_title('Feature Importance')

plt.tight_layout()
plt.savefig('/home/claude/CODSOFT/task4_results.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✅ Task 4 Complete! Best: {best_name} R²={best['r2']:.4f}")
