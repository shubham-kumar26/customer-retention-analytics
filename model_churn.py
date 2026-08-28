import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

engine = create_engine('postgresql://postgres:1234@localhost:5432/olist_churn')

# Pull customer-level features
query = """
SELECT 
    c.customer_unique_id,
    COUNT(DISTINCT o.order_id) AS total_orders,
    SUM(op.payment_value) AS total_spent,
    AVG(op.payment_value) AS avg_order_value,
    AVG(r.review_score) AS avg_review_score,
    AVG(EXTRACT(DAY FROM (o.order_delivered_customer_date - o.order_estimated_delivery_date))) AS avg_delivery_delay,
    MODE() WITHIN GROUP (ORDER BY op.payment_type) AS most_common_payment_type,
    AVG(op.payment_installments) AS avg_installments
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN order_payments op ON o.order_id = op.order_id
LEFT JOIN order_reviews r ON o.order_id = r.order_id
WHERE o.order_status = 'delivered'
GROUP BY c.customer_unique_id
"""

df = pd.read_sql(query, engine)
print(df.shape)
print(df.head())

# Target: is this customer a repeat buyer? (more than 1 order)
df['is_repeat_customer'] = (df['total_orders'] > 1).astype(int)

print("\nTarget distribution:")
print(df['is_repeat_customer'].value_counts())

# Handle missing values (some customers have no review or no delivery delay data)
df['avg_review_score'] = df['avg_review_score'].fillna(df['avg_review_score'].mean())
df['avg_delivery_delay'] = df['avg_delivery_delay'].fillna(df['avg_delivery_delay'].mean())

# Encode payment type
df['payment_type_encoded'] = df['most_common_payment_type'].astype('category').cat.codes

# Select features and target
features = ['total_spent', 'avg_order_value', 'avg_review_score', 'avg_delivery_delay', 'payment_type_encoded', 'avg_installments']
X = df[features]
y = df['is_repeat_customer']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression baseline
lr_model = LogisticRegression(class_weight='balanced', max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

print("\n--- Logistic Regression ---")
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("\n--- Random Forest ---")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# XGBoost
from xgboost import XGBClassifier

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = XGBClassifier(n_estimators=100, scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

print("\n--- XGBoost ---")
print(confusion_matrix(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))

# ROC-AUC for all three models
y_prob_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

print("\n--- ROC-AUC Scores ---")
print("Logistic Regression:", roc_auc_score(y_test, y_prob_lr))
print("Random Forest:", roc_auc_score(y_test, y_prob_rf))
print("XGBoost:", roc_auc_score(y_test, y_prob_xgb))

# Feature importance
importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
print("\n--- Feature Importance (Random Forest) ---")
print(importances)

# ============================================
# Customer Segmentation
# ============================================
df['spend_tier'] = pd.qcut(df['total_spent'], q=3, labels=['Low', 'Medium', 'High'])
df['predicted_repeat_prob'] = rf_model.predict_proba(df[features])[:, 1]

segment_summary = df.groupby('spend_tier').agg(
    avg_total_spent=('total_spent', 'mean'),
    avg_order_value=('avg_order_value', 'mean'),
    avg_repeat_prob=('predicted_repeat_prob', 'mean'),
    customer_count=('customer_unique_id', 'count')
).reset_index()

print("\n--- Segment Summary ---")
print(segment_summary)

# ============================================
# A/B Test Simulation: Retention Offer Impact
# ============================================
from scipy import stats

np.random.seed(42)
at_risk_customers = df[(df['is_repeat_customer'] == 0) & (df['spend_tier'] == 'High')].copy()

# Split into control (no offer) and treatment (hypothetical 10% discount offer)
control = at_risk_customers.sample(frac=0.5, random_state=1)
treatment = at_risk_customers.drop(control.index)

# Simulate outcome: baseline 5% repeat rate vs 8% with a retention offer
control_outcomes = np.random.binomial(1, 0.05, size=len(control))
treatment_outcomes = np.random.binomial(1, 0.08, size=len(treatment))

t_stat, p_value = stats.ttest_ind(control_outcomes, treatment_outcomes)

print("\n--- A/B Test: Retention Offer Simulation (High-value, one-time customers) ---")
print(f"Control group size: {len(control)}, repeat rate: {control_outcomes.mean():.2%}")
print(f"Treatment group size: {len(treatment)}, repeat rate: {treatment_outcomes.mean():.2%}")
print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")
print("Result:", "Statistically significant improvement" if p_value < 0.05 else "Not statistically significant — larger sample or stronger effect needed")
