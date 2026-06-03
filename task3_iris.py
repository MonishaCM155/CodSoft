import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("TASK 3: IRIS FLOWER CLASSIFICATION")
print("=" * 60)

df = pd.read_csv('/home/claude/datasets/IRIS.csv')
print(f"Dataset shape: {df.shape}")
print(df['species'].value_counts())

# EDA - pairplot style
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Iris Flower EDA', fontsize=16, fontweight='bold')
colors = {'Iris-setosa':'#e74c3c','Iris-versicolor':'#3498db','Iris-virginica':'#2ecc71'}
pairs = [('sepal_length','sepal_width'),('petal_length','petal_width'),('sepal_length','petal_length')]
for i, (x, y) in enumerate(pairs):
    for species, color in colors.items():
        sub = df[df['species']==species]
        axes[0,i].scatter(sub[x], sub[y], label=species, color=color, alpha=0.7)
    axes[0,i].set_xlabel(x); axes[0,i].set_ylabel(y); axes[0,i].legend(fontsize=7)
    axes[0,i].set_title(f'{x} vs {y}')

for i, col in enumerate(['sepal_length','sepal_width','petal_length']):
    for species, color in colors.items():
        df[df['species']==species][col].plot(kind='hist', ax=axes[1,i], alpha=0.6, color=color, label=species, bins=15)
    axes[1,i].set_title(f'{col} Distribution'); axes[1,i].legend(fontsize=7)

plt.tight_layout()
plt.savefig('/home/claude/CODSOFT/task3_eda.png', dpi=150, bbox_inches='tight')
plt.close()

le = LabelEncoder()
X = df.drop('species', axis=1)
y = le.fit_transform(df['species'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42)
}
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = {'preds': preds, 'accuracy': acc, 'model': model}
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=le.classes_))

best_name = max(results, key=lambda k: results[k]['accuracy'])
best = results[best_name]
cm = confusion_matrix(y_test, best['preds'])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f'Best Model: {best_name} (Accuracy: {best["accuracy"]:.4f})', fontsize=14)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[0],
            xticklabels=le.classes_, yticklabels=le.classes_)
axes[0].set_title('Confusion Matrix')

acc_df = pd.Series({k: v['accuracy'] for k,v in results.items()}).sort_values()
acc_df.plot(kind='barh', ax=axes[1], color=['#e74c3c','#f39c12','#3498db','#2ecc71'])
axes[1].set_title('Model Accuracy Comparison'); axes[1].set_xlim(0.9, 1.01)
for i, v in enumerate(acc_df):
    axes[1].text(v+0.001, i, f'{v:.4f}', va='center')

plt.tight_layout()
plt.savefig('/home/claude/CODSOFT/task3_results.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✅ Task 3 Complete! Best: {best_name} Accuracy={best['accuracy']:.4f}")
