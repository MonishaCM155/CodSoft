import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("TASK 1: TITANIC SURVIVAL PREDICTION")
print("=" * 60)

df = pd.read_csv('/home/claude/datasets/Titanic-Dataset.csv')
print(f"Dataset shape: {df.shape}")

# EDA Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('Titanic Survival EDA', fontsize=16, fontweight='bold')
df['Survived'].value_counts().plot(kind='bar', ax=axes[0,0], color=['#e74c3c','#2ecc71'])
axes[0,0].set_title('Survival Count'); axes[0,0].set_xticklabels(['Died','Survived'], rotation=0)
df.groupby('Sex')['Survived'].mean().plot(kind='bar', ax=axes[0,1], color=['#3498db','#e91e63'])
axes[0,1].set_title('Survival Rate by Sex'); axes[0,1].set_xticklabels(axes[0,1].get_xticklabels(), rotation=0)
df['Age'].dropna().plot(kind='hist', ax=axes[1,0], bins=30, color='#9b59b6', edgecolor='white')
axes[1,0].set_title('Age Distribution')
df.groupby('Pclass')['Survived'].mean().plot(kind='bar', ax=axes[1,1], color=['#f39c12','#27ae60','#e74c3c'])
axes[1,1].set_title('Survival Rate by Class'); axes[1,1].set_xticklabels(['1st','2nd','3rd'], rotation=0)
plt.tight_layout()
plt.savefig('/home/claude/CODSOFT/task1_eda.png', dpi=150, bbox_inches='tight')
plt.close()

# Preprocessing - use assignment not inplace (pandas CoW)
df = df.drop(['Cabin','Name','Ticket','PassengerId'], axis=1)
df = df.assign(
    Age=df['Age'].fillna(df['Age'].median()),
    Embarked=df['Embarked'].fillna(df['Embarked'].mode()[0]),
    Fare=df['Fare'].fillna(df['Fare'].median())
)
le = LabelEncoder()
df = df.assign(
    Sex=le.fit_transform(df['Sex']),
    Embarked=le.fit_transform(df['Embarked']),
    FamilySize=df['SibSp'] + df['Parch'] + 1
)
df = df.assign(IsAlone=(df['FamilySize'] == 1).astype(int))

print("Missing after preprocessing:", df.isnull().sum().sum())

X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = {'model': model, 'preds': preds, 'accuracy': acc}
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=['Died','Survived']))

best_name = max(results, key=lambda k: results[k]['accuracy'])
best = results[best_name]
cm = confusion_matrix(y_test, best['preds'])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f'Best Model: {best_name} (Accuracy: {best["accuracy"]:.4f})', fontsize=14)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Died','Survived'], yticklabels=['Died','Survived'])
axes[0].set_title('Confusion Matrix'); axes[0].set_ylabel('Actual'); axes[0].set_xlabel('Predicted')
rf = results['Random Forest']['model']
feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
feat_imp.plot(kind='barh', ax=axes[1], color='#3498db')
axes[1].set_title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.savefig('/home/claude/CODSOFT/task1_results.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✅ Task 1 Complete! Best: {best_name} Accuracy={best['accuracy']:.4f}")
