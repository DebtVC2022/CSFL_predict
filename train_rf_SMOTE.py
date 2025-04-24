import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, brier_score_loss
from sklearn.utils import compute_class_weight
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from imblearn.over_sampling import ADASYN, SMOTE
from skopt import BayesSearchCV
import shap
import lime
import lime.lime_tabular
import os

# 设置全部图片字体为times new roman
plt.rcParams['font.family'] = 'Times New Roman'

# ========== 数据加载与预处理 ==========
excel_file = "data_20250123.xlsx"
df = pd.read_excel(excel_file)
X = df.iloc[:, 1:].fillna(-99)
y = df.iloc[:, 0]

scaler = preprocessing.MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ========== 类别不平衡处理 ==========
sampler = SMOTE(random_state=42)
# sampler = ADASYN(random_state=42)
X_resampled, y_resampled = sampler.fit_resample(X_scaled, y)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_resampled), y=y_resampled)
sample_weights = np.array([class_weights[i] for i in y_resampled])

# 数据划分
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X_resampled, y_resampled, sample_weights, test_size=0.2, random_state=42
)

# ========== 模型训练 ==========
model = RandomForestClassifier(random_state=42)
param_space = {
    'n_estimators': [10, 30, 50, 70, 100],
    'max_features': ['sqrt', 'log2', 3],
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 3, 5],
    'min_samples_leaf': [1, 2, 4],
    'n_jobs': [-1]
}

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

# ========== 性能评估 ==========
y_test_pred = calibrated_model.predict(X_test)
y_test_proba = calibrated_model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_test_proba)
brier = brier_score_loss(y_test, y_test_proba)



# ========== ROC 曲线 ==========
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC={auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("ROC Curve")
plt.legend(); plt.tight_layout()
plt.savefig("./rf_img/revisions/rf_auc_curve_revisions_SMOTE.png")
plt.show()

# ========== 校准曲线 ==========
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
plt.savefig("./rf_img/revisions/rf_calibration_compare_1_revisions_SMOTE.png")
plt.show()

# ========== 特征重要性 ==========
importance_scores = best_model.feature_importances_
sorted_idx = np.argsort(importance_scores)[::-1]
sorted_features = np.array(df.columns[1:])[sorted_idx]
sorted_importance = importance_scores[sorted_idx]

plt.figure(figsize=(12, 8))
plt.barh(sorted_features, sorted_importance, color=plt.cm.viridis(np.linspace(0, 1, len(sorted_features))))
plt.xlabel("Feature Importance")
plt.title("Feature Importance (RF)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("./rf_img/revisions/rf_feature_importance_revisions_SMOTE.png")
plt.show()

# ========== SHAP 可解释性 ========== #
print("\n生成 SHAP 图...")
shap.initjs()
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)
print(shap_values)

plt.figure(figsize=(12, 8))  # 显式创建新图
if isinstance(shap_values, list) and len(shap_values) == 2:
    shap.summary_plot(shap_values[1], X_test, feature_names=df.columns[1:].tolist(), show=False)
else:
    shap.summary_plot(shap_values, X_test, feature_names=df.columns[1:].tolist(), show=False)
plt.tight_layout()
plt.savefig("./rf_img/revisions/rf_shap_summary_revisions_SMOTE.png", dpi=300)
plt.close()

# ========== LIME 可解释性 ==========
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
fig.savefig("./rf_img/revisions/rf_lime_explanation_0_revisions_SMOTE.png")

print("\n在测试集上的性能评估:")
print(f"准确率: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"精确度: {precision_score(y_test, y_test_pred):.4f}")
print(f"召回率: {recall_score(y_test, y_test_pred):.4f}")
print(f"F1分数: {f1_score(y_test, y_test_pred):.4f}")
print(f"AUC: {auc:.4f}")
print(f"Brier分数: {brier:.4f}")

# ========== 模型保存 ==========
joblib.dump(calibrated_model, "./model_best/best_model_rf_revisions_SMOTE.pkl")
print("模型已保存到 ./model_best/best_model_rf_revisions_SMOTE.pkl")
