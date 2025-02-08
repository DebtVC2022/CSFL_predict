import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, roc_curve
)
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from skopt import BayesSearchCV
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 读取数据
excel_file = "data_20250123.xlsx"
df = pd.read_excel(excel_file)
X = df.iloc[:, 1:].fillna(-99)  # 填充缺失值
y = df.iloc[:, 0]

# 数据标准化
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# SMOTE 上采样
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 贝叶斯搜索参数空间
param_space = {
    'C': (1e-3, 1e3, 'log-uniform'),
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear'],
    'max_iter': (50, 1000),
    'class_weight': ['balanced']
}

# 定义逻辑回归模型
model = LogisticRegression()

# 贝叶斯优化
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

# 计算并绘制校准曲线
fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_test_pred_proba, n_bins=10)
plt.figure(figsize=(8, 6))
plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Logistic Regression")
plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
plt.xlabel("Mean predicted value")
plt.ylabel("Fraction of positives")
plt.title("Calibration Curve")
plt.legend()
plt.savefig("./logistic_img/logistic_calibration_curve.png")
plt.show()


# 绘制ROC曲线
main_color = plt.cm.plasma(0.3)
fill_color = plt.cm.plasma(0.2)
chance_color = "#FFD700" #plt.cm.plasma(0.1)



fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba)
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color=main_color, lw=3, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], color=chance_color, lw=2, linestyle='--', label='Chance Level')
plt.fill_between(fpr, tpr, alpha=0.2, color=fill_color)  # 添加填充颜色
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=15, weight='bold')
plt.ylabel('True Positive Rate (TPR)', fontsize=15, weight='bold')
plt.title('ROC Curve', fontsize=20, weight='bold')
plt.legend(loc='lower right', fontsize=12)
plt.grid(linestyle='--', linewidth=0.6, alpha=0.7)  # 设置背景网格样式
plt.tight_layout()
plt.style.use('seaborn-darkgrid')
plt.savefig("./logistic_img/logistic_auc_curve.png")



# 保存最优模型
best_model_file = "./model_best/best_model_logistic.pkl"
joblib.dump(calibrated_model, best_model_file)
print(f"最优模型已保存到 {best_model_file}")

# 特征重要性分析
coefficients = best_model.coef_.flatten()
coef_df = pd.DataFrame({'Feature': df.columns[1:], 'Coefficient': coefficients})
coef_df['Importance'] = np.abs(coef_df['Coefficient'])
coef_df = coef_df.sort_values(by='Importance', ascending=False)

# 可视化特征重要性
plt.figure(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0, 1, len(coef_df)))
plt.barh(coef_df['Feature'][::-1], coef_df['Coefficient'][::-1], color=colors, edgecolor='black')
plt.title('Feature Importance', fontsize=16)
plt.xlabel('Coefficient Value', fontsize=14)
plt.ylabel('Features', fontsize=14)
plt.tight_layout()
plt.savefig("./logistic_img/logistic_feature_importance.png")
plt.show()