# 完整版 XGBoost：添加类别不平衡方法（ADASYN + 权重）、校准（Platt/Isotonic）、SHAP+LIME解释
import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import preprocessing  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, brier_score_loss
import matplotlib.pyplot as plt
from imblearn.over_sampling import ADASYN, SMOTE
from skopt import BayesSearchCV
from xgboost import XGBClassifier
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.utils import compute_class_weight
import shap
import lime
import lime.lime_tabular
import seaborn as sns

# 设置全部图片字体为times new roman
plt.rcParams['font.family'] = 'Times New Roman'

# ========== 数据读取与预处理 ==========
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

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X_resampled, y_resampled, sample_weights, test_size=0.2, random_state=42
)

# ========== 模型训练与调参 ==========
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
param_space = {
    'n_estimators': [30, 50, 80],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_child_weight': [2, 4, 6],
    'scale_pos_weight': [1, 2, 4],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.5],
    'reg_alpha': [0, 0.1, 1, 5],
    'reg_lambda': [0, 1, 5, 10],
    'n_jobs': [-1]
}

bayes_search = BayesSearchCV(
    estimator=model,
    search_spaces=param_space,
    cv=StratifiedKFold(n_splits=5),
    scoring='f1_macro',
    n_jobs=-1,
    random_state=42
)
bayes_search.fit(X_train, y_train, sample_weight=w_train)
best_model = bayes_search.best_estimator_
print("最佳参数:", bayes_search.best_params_)

# ========== 性能评估 ==========
y_test_pred = best_model.predict(X_test)
y_test_proba = best_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_test_proba)


# ========== 校准曲线对比 ==========
calibrators = {
    "Platt": CalibratedClassifierCV(best_model, method='sigmoid', cv=5),
    "Isotonic": CalibratedClassifierCV(best_model, method='isotonic', cv=5)
}
calibrated = {}
for name, calibrator in calibrators.items():
    calibrator.fit(X_train, y_train)
    proba = calibrator.predict_proba(X_test)[:, 1]
    calibrated[name] = (proba, brier_score_loss(y_test, proba))
    print(f"{name} Brier 分数: {calibrated[name][1]:.4f}")

brier_orig = brier_score_loss(y_test, y_test_proba)
print(f"未校准 Brier 分数: {brier_orig:.4f}")

plt.figure()
frac, mean = calibration_curve(y_test, y_test_proba, n_bins=10)
plt.plot(mean, frac, label='Uncalibrated')
for name, (proba, _) in calibrated.items():
    frac, mean = calibration_curve(y_test, proba, n_bins=10)
    plt.plot(mean, frac, label=name)
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
plt.xlabel("Mean predicted value"); plt.ylabel("Fraction of positives")
plt.title("Calibration Curve")
plt.legend(); plt.tight_layout()
plt.savefig("./xgb_img/revisions/xgb_calibration_compare_1_revisions_SMOTE.png")
plt.show()

# ========== SHAP 解释 ==========
explainer = shap.Explainer(best_model, X_train)
# explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)
if isinstance(shap_values, list):  # 二分类情况处理
    shap_values = shap_values[1]

plt.figure(figsize=(12, 8))  # 新建一个图形对象
shap.summary_plot(shap_values, features=X_test, feature_names=X.columns.tolist(), show=False)
plt.tight_layout()
plt.savefig("./xgb_img/revisions/xgb_shap_summary_1_revisions_SMOTE.png")
plt.close()

# ========== LIME 单样本解释 ==========
lime_exp = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    feature_names=X.columns.tolist(),
    class_names=['Negative', 'Positive'],
    mode='classification'
)
exp = lime_exp.explain_instance(X_test[0], best_model.predict_proba, num_features=10)
fig = exp.as_pyplot_figure()
fig.tight_layout()
fig.savefig("./xgb_img/revisions/xgb_lime_explanation_0_1_revisions_SMOTE.png")

# ========== ROC 曲线 ==========
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC={auc:.2f}", lw=2)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("ROC Curve")
plt.legend(); plt.tight_layout()
plt.savefig("./xgb_img/revisions/xgb_auc_curve_1_revisions_SMOTE.png")
plt.show()

# ========== 特征重要性图 ==========
importance_scores = best_model.feature_importances_
sorted_indices = np.argsort(importance_scores)[::-1]
sorted_features = np.array(X.columns)[sorted_indices]
sorted_importance = importance_scores[sorted_indices]

plt.figure(figsize=(12, 8))
plt.barh(sorted_features, sorted_importance, color=plt.cm.viridis(np.linspace(0, 1, len(sorted_features))))
plt.xlabel("Feature Importance")
plt.title("XGBoost Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("./xgb_img/revisions/xgb_feature_importance_1_revisions_SMOTE.png")
plt.show()


print("\n性能评估:")
print(f"准确率: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"精确度: {precision_score(y_test, y_test_pred):.4f}")
print(f"召回率: {recall_score(y_test, y_test_pred):.4f}")
print(f"F1分数: {f1_score(y_test, y_test_pred):.4f}")
print(f"AUC分数: {auc:.4f}")

# ========== 模型保存 ==========
joblib.dump(best_model, "./model_best/best_model_xgb_1_revisions_SMOTE.pkl")
print("模型已保存")
