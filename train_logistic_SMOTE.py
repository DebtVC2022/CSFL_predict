import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.utils import compute_class_weight
from imblearn.over_sampling import SMOTE, ADASYN
from skopt import BayesSearchCV

import shap
import lime
import lime.lime_tabular

# 设置全部图片字体为times new roman
plt.rcParams['font.family'] = 'Times New Roman'

# 读取数据
excel_file = "data_20250123.xlsx"
df = pd.read_excel(excel_file)
X = df.iloc[:, 1:].fillna(-99)
y = df.iloc[:, 0]

# 数据归一化
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 类别不平衡处理（ADASYN）
sampler = SMOTE(random_state=42)
# sampler = ADASYN(random_state=42)
X_resampled, y_resampled = sampler.fit_resample(X_scaled, y)

# 样本权重
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_resampled), y=y_resampled)
sample_weights = np.array([class_weights[i] for i in y_resampled])

# 划分数据
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X_resampled, y_resampled, sample_weights, test_size=0.2, random_state=42
)

# 贝叶斯参数空间
param_space = {
    'C': (1e-3, 1e3, 'log-uniform'),
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear'],
    'max_iter': (100, 1000),
    'class_weight': ['balanced']
}

model = LogisticRegression()

bayes_search = BayesSearchCV(
    estimator=model,
    search_spaces=param_space,
    cv=StratifiedKFold(n_splits=3),
    scoring='f1_macro',
    n_jobs=-1,
    random_state=42
)

bayes_search.fit(X_train, y_train, sample_weight=w_train)
best_model = bayes_search.best_estimator_
print("最佳参数:", bayes_search.best_params_)

# 校准模型
calibrated_model = CalibratedClassifierCV(base_estimator=best_model, method='sigmoid', cv=5)
calibrated_model.fit(X_train, y_train)

# 预测
y_test_pred = calibrated_model.predict(X_test)
y_test_proba = calibrated_model.predict_proba(X_test)[:, 1]



# 相关性热图
plt.figure(figsize=(16, 16))
sns.heatmap(df.iloc[:, 1:].corr(), annot=True, vmax=1, square=True, cmap="Blues", fmt='.2g')
plt.title('Correlation heatmap', weight='bold')
plt.savefig('./logistic_img/revisions/df_corr_revisions.png', bbox_inches='tight', transparent=True)
plt.close()

# 校准模型
calibrators = {
    "Platt": CalibratedClassifierCV(best_model, method='sigmoid', cv=5),
    "Isotonic": CalibratedClassifierCV(best_model, method='isotonic', cv=5)
}
calibrated = {}

for name, calibrator in calibrators.items():
    calibrator.fit(X_train, y_train)
    proba = calibrator.predict_proba(X_test)[:, 1]
    calibrated[name] = (proba, brier_score_loss(y_test, proba))
    print(f"{name} Brier分数: {calibrated[name][1]:.4f}")

brier_orig = brier_score_loss(y_test, y_test_proba)
print(f"原始 Brier 分数: {brier_orig:.4f}")

# 校准曲线
plt.figure(figsize=(8, 6))
frac, mean = calibration_curve(y_test, y_test_proba, n_bins=10)
plt.plot(mean, frac, 's--', label='Uncalibrated')
for name, (proba, _) in calibrated.items():
    frac, mean = calibration_curve(y_test, proba, n_bins=10)
    plt.plot(mean, frac, 'o-', label=name)
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
plt.xlabel("Mean predicted value")
plt.ylabel("Fraction of positives")
plt.title("Calibration Curve")
plt.legend(); plt.tight_layout()
plt.savefig("./logistic_img/revisions/logistic_calibration_compare_1_revisions_SMOTE.png")
plt.show()
# ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='blue', lw=3, label=f'AUC = {roc_auc_score(y_test, y_test_proba):.2f}')
plt.plot([0, 1], [0, 1], color='#FFD700', lw=2, linestyle='--', label='Chance Level')
plt.fill_between(fpr, tpr, alpha=0.2, color='skyblue')
plt.xlabel('FPR', fontsize=15)
plt.ylabel('TPR', fontsize=15)
plt.title('ROC Curve', fontsize=18)
plt.legend(loc='lower right')
plt.grid(linestyle='--', linewidth=0.6, alpha=0.7)
plt.tight_layout()
plt.savefig("./logistic_img/revisions/logistic_auc_curve_revisions_SMOTE.png")
plt.close()

# 特征重要性（系数）
coef = best_model.coef_.flatten()
coef_df = pd.DataFrame({'Feature': df.columns[1:], 'Coefficient': coef})
coef_df['Importance'] = np.abs(coef_df['Coefficient'])
coef_df = coef_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0, 1, len(coef_df)))
plt.barh(coef_df['Feature'][::-1], coef_df['Coefficient'][::-1], color=colors, edgecolor='black')
plt.xlabel("Coefficient Value")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("./logistic_img/revisions/logistic_feature_importance_revisions_SMOTE.png")
plt.close()

# SHAP 可解释性
print("\n生成 SHAP 图...")
shap.initjs()
explainer = shap.Explainer(best_model, X_train)
shap_values = explainer(X_test)
plt.figure(figsize=(12, 8))  # 新建一个图形对象
shap.summary_plot(shap_values, X_test, feature_names=df.columns[1:].tolist(), show=False)
plt.tight_layout()
plt.savefig("./logistic_img/revisions/logistic_shap_summary_revisions_SMOTE.png")
plt.close()

# LIME 可解释性
print("\nLIME 单样本解释...")
lime_exp = lime.lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=df.columns[1:].tolist(),
    class_names=['Negative', 'Positive'],
    discretize_continuous=True,
    mode='classification'
)
exp = lime_exp.explain_instance(X_test[0], calibrated_model.predict_proba, num_features=10)
fig = exp.as_pyplot_figure()
fig.tight_layout()
fig.savefig("./logistic_img/revisions/logistic_lime_explanation_0_revisions_SMOTE.png")

# 性能评估
print("\n在测试集上的性能评估:")
print(f"准确率: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"精确度: {precision_score(y_test, y_test_pred):.4f}")
print(f"召回率: {recall_score(y_test, y_test_pred):.4f}")
print(f"F1分数: {f1_score(y_test, y_test_pred):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_test_proba):.4f}")

# 模型保存
joblib.dump(calibrated_model, "./model_best/best_model_logistic_revisions_SMOTE.pkl")
print("模型已保存到 ./model_best/best_model_logistic_revisions_SMOTE.pkl")
