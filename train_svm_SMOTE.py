import os
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn import preprocessing  
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, brier_score_loss, log_loss
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

from imblearn.over_sampling import SMOTE, ADASYN
from skopt import BayesSearchCV

import shap
import lime
import lime.lime_tabular

# 设置全部图片字体为times new roman
plt.rcParams['font.family'] = 'Times New Roman'

# ========== 数据读取与预处理 ==========
df = pd.read_excel("data_20250123.xlsx")
X = df.iloc[:, 1:].fillna(-99)
y = df.iloc[:, 0]

scaler = preprocessing.MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ========== 类别不平衡方法：选择 SMOTE 或 ADASYN ==========
sampler = SMOTE(random_state=42)
# sampler = ADASYN(random_state=42)
X_resampled, y_resampled = sampler.fit_resample(X_scaled, y)
print(f"当前使用的类别不平衡方法：{sampler.__class__.__name__}")

# ========== 数据划分 ==========
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# ========== SVM 模型与调参 ==========
model = SVC(class_weight='balanced', probability=True)

param_space = {
    'C': (1e-3, 1e3, 'log-uniform'),
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto']
}

bayes_search = BayesSearchCV(
    estimator=model,
    search_spaces=param_space,
    cv=StratifiedKFold(n_splits=5),
    scoring='f1_macro',
    n_jobs=-1,
    random_state=42
)
bayes_search.fit(X_train, y_train)
best_model = bayes_search.best_estimator_
print(f"最佳参数: {bayes_search.best_params_}")

# ========== 性能与校准 ==========
y_test_pred = best_model.predict(X_test)
y_test_proba = best_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_test_proba)

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
plt.xlabel("Mean predicted value"); plt.ylabel("Fraction of positives")
plt.title("Calibration Curve")
plt.legend(); plt.tight_layout()
plt.savefig("./svm_img/revisions/svm_calibration_compare_1_revisions_SMOTE.png")
plt.show()

# ========== SHAP 可解释性 ==========
explainer = shap.KernelExplainer(best_model.predict_proba, X_train[:100])
shap_values = explainer.shap_values(X_test[:50])
plt.figure(figsize=(12, 8))  # 新建一个图形对象
shap.summary_plot(shap_values[1], features=X_test[:50], feature_names=df.columns[1:].tolist(), show=False)
plt.tight_layout()
plt.savefig("./svm_img/revisions/svm_shap_summary_1_revisions_SMOTE.png")
plt.close()

# ========== LIME 可解释性 ==========
lime_exp = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    feature_names=df.columns[1:].tolist(),
    class_names=['Negative', 'Positive'],
    mode='classification'
)

exp = lime_exp.explain_instance(X_test[0], best_model.predict_proba, num_features=10)
fig = exp.as_pyplot_figure()
fig.tight_layout()
fig.savefig("./svm_img/revisions/svm_lime_explanation_0_1_revisions_SMOTE.png")

# ========== 性能评估 ==========
acc = accuracy_score(y_test, y_test_pred)
prec = precision_score(y_test, y_test_pred)
rec = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
auc = roc_auc_score(y_test, y_test_proba)
brier = brier_score_loss(y_test, y_test_proba)
logloss = log_loss(y_test, y_test_proba)

# ========== ROC 曲线 ==========
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}", lw=2)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("SVM ROC Curve (ADASYN)")
plt.legend(); plt.tight_layout()
plt.savefig("./svm_img/revisions/svm_auc_curve_1_revisions_SMOTE.png")
plt.show()

# ========== 特征重要性图（基于支持向量） ==========
support_vectors = best_model.support_vectors_
dual_coef = best_model.dual_coef_.flatten()
feature_importance = np.abs(np.dot(dual_coef, support_vectors))

# 归一化
feature_importance = (feature_importance - feature_importance.min()) / (feature_importance.max() - feature_importance.min())
sorted_idx = np.argsort(feature_importance)[::-1]
sorted_features = np.array(df.columns[1:])[sorted_idx]
sorted_importance = feature_importance[sorted_idx]

plt.figure(figsize=(12, 8))
plt.barh(sorted_features, sorted_importance, color=plt.cm.viridis(np.linspace(0, 1, len(sorted_features))))
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("SVM Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("./svm_img/revisions/svm_feature_importance_1_revisions_SMOTE.png")
plt.show()

print("\n在测试集上的性能评估:")
print(f"准确率: {acc:.4f}")
print(f"精确率: {prec:.4f}")
print(f"召回率: {rec:.4f}")
print(f"F1分数: {f1:.4f}")
print(f"AUC分数: {auc:.4f}")

# ========== 模型保存 ==========
model_path = "./model_best/best_model_svm_1_revisions_SMOTE.pkl"
joblib.dump(best_model, model_path)
print(f"\n模型已保存至: {model_path}")
