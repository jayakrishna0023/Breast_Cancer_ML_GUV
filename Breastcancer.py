import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

data = pd.read_csv('BreastCancer.csv')


print("Dataset Shape:", data.shape)
print("Data Types:\n", data.dtypes)
print("Missing Values:\n", data.isnull().sum())
print("Target Distribution:\n", data['diagnosis'].value_counts())

sns.countplot(data['diagnosis'])
plt.title('Diagnosis Distribution')
plt.show()

data['diagnosis'] = LabelEncoder().fit_transform(data['diagnosis'])

plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), cmap='coolwarm', annot=False)
plt.title('Correlation Matrix')
plt.show()

X = data.drop(columns=['diagnosis'])
y = data['diagnosis']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

rf_model = RandomForestClassifier(random_state=42)
gb_model = GradientBoostingClassifier(random_state=42)
ada_model = AdaBoostClassifier(random_state=42)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
lgbm_model = LGBMClassifier(random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

rf_grid = GridSearchCV(rf_model, param_grid, cv=StratifiedKFold(5), scoring='f1', verbose=1)
rf_grid.fit(X_train, y_train)

print("Best Parameters for Random Forest:", rf_grid.best_params_)

final_rf_model = rf_grid.best_estimator_
final_rf_model.fit(X_train, y_train)

y_pred = final_rf_model.predict(X_test)
y_pred_prob = final_rf_model.predict_proba(X_test)[:, 1]


print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_prob))

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc_score(y_test, y_pred_prob):.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
plt.plot(recall, precision, label='Random Forest')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

stacking_model = StackingClassifier(
    estimators=[('rf', rf_model), ('gb', gb_model), ('ada', ada_model), ('xgb', xgb_model), ('lgbm', lgbm_model)],
    final_estimator=GradientBoostingClassifier(random_state=42),
    cv=StratifiedKFold(5)
)

stacking_model.fit(X_train, y_train)
stacking_pred = stacking_model.predict(X_test)
stacking_pred_prob = stacking_model.predict_proba(X_test)[:, 1]

print("Stacking Classifier Report:\n", classification_report(y_test, stacking_pred))
print("Stacking ROC-AUC Score:", roc_auc_score(y_test, stacking_pred_prob))

importances = final_rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(10, 6))
plt.title('Feature Importances (Random Forest)')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), features[indices], rotation=90)
plt.tight_layout()
plt.show()
