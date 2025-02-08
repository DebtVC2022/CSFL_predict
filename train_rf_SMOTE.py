import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from skopt import BayesSearchCV
from sklearn.calibration import CalibratedClassifierCV

# 读取数据
excel_file = "data_20250123.xlsx"
df = pd.read_excel(excel_file)
X = df.iloc[:, 1:]
X = X.fillna(-99)
y = df.iloc[:, 0]

# 数据标准化
scaler = preprocessing.MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# SMOTE 上采样
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 定义随机森林模型
model = RandomForestClassifier(random_state=42)

# 贝叶斯搜索参数空间
param_space = {
    'n_estimators': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 60],
    'max_features': ['sqrt', 2, 3, 'log2'],
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'min_samples_split': [2, 3, 4, 5, 6],
    'min_samples_leaf': [1, 2, 3, 4, 5, 6],
    'n_jobs':[-1]
}

# 贝叶斯搜索
bayes_search = BayesSearchCV(
    estimator=model,
    search_spaces=param_space,
    cv=StratifiedKFold(n_splits=5),
    scoring='f1_macro',
    n_jobs=-1,
    random_state=42
)

# 模型训练
bayes_search.fit(X_train, y_train)

# 获取最佳模型
best_model = bayes_search.best_estimator_
print(f"最佳参数: {bayes_search.best_params_}")

# 校准模型
calibrated_model = CalibratedClassifierCV(base_estimator=best_model, method='sigmoid')
calibrated_model.fit(X_train, y_train)

# 测试集预测
y_test_pred = calibrated_model.predict(X_test)
y_test_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]

# 计算评价指标
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
auc = roc_auc_score(y_test, y_test_pred_proba)

print("在测试集上的性能评估:")
print(f"准确率: {accuracy}")
print(f"精确度: {precision}")
print(f"召回率: {recall}")
print(f"F1分数: {f1}")
print(f"AUC分数: {auc}")

# 绘制 ROC 曲线
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba)
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='blue', lw=3, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Chance Level')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(linestyle='--', linewidth=0.6, alpha=0.7)
plt.tight_layout()
plt.savefig("./rf_img/rf_auc_curve.png")
plt.show()

# 绘制校准曲线
from sklearn.calibration import calibration_curve
fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_test_pred_proba, n_bins=10)
plt.figure(figsize=(8, 6))
plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Random Forest")
plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
plt.xlabel("Mean predicted value")
plt.ylabel("Fraction of positives")
plt.title("Calibration Curve")
plt.legend()
plt.tight_layout()
plt.savefig("./rf_img/rf_calibration_curve.png")
plt.show()

# 绘制特征重要性图
importance_scores = best_model.feature_importances_
sorted_indices = np.argsort(importance_scores)[::-1]
sorted_features = np.array(df.columns[1:])[sorted_indices]
sorted_importance = importance_scores[sorted_indices]

plt.figure(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_features)))
plt.barh(sorted_features, sorted_importance, color=colors, edgecolor='black')
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("./rf_img/rf_feature_importance.png")
plt.show()

# 保存最优模型
best_model_file = "./model_best/best_model_rf.pkl"
joblib.dump(calibrated_model, best_model_file)
print(f"最优模型已保存到 {best_model_file}")
