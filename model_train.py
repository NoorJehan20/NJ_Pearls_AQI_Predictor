import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Loading Dataset
df = pd.read_csv("feat_eng.csv")

# Drop only PM2.5 & PM10, keep lag features
df = df.drop(columns=['pm2p5', 'pm10'], errors='ignore')
print("Dropped features: ['pm2p5', 'pm10']")

# Define features & target
features = [col for col in df.columns if col not in ['time', 'datetime', 'aqi']]
target = 'aqi'
X = df[features]
y = df[target]

# Correlation Check 
print("\nChecking feature correlations...")
corr = df[features + [target]].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Drop highly correlated features
threshold = 0.9
to_drop = set()
for i in range(len(corr.columns)):
    for j in range(i):
        if abs(corr.iloc[i, j]) > threshold:
            to_drop.add(corr.columns[i])
if to_drop:
    print(f"Dropping highly correlated features: {to_drop}")
    features = [f for f in features if f not in to_drop]
X = X[features]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Data split: {X_train.shape[0]} train rows, {X_test.shape[0]} test rows")

# BASELINE MODELS
os.makedirs("saved_models", exist_ok=True)
results = {}

print("\nTraining Ridge (baseline)...")
ridge = Ridge()
ridge.fit(X_train, y_train)
results['Ridge'] = {'model': ridge, 'preds': ridge.predict(X_test)}

#RANDOM FOREST TUNING 
print("\n🔍 Tuning Random Forest...")
rf_params = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}
rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    rf_params, n_iter=15, scoring='r2', cv=3, n_jobs=-1, random_state=42
)
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_
print(f"Best RF Params: {rf_search.best_params_}")
results['Random Forest'] = {'model': best_rf, 'preds': best_rf.predict(X_test)}

# XGBOOST TUNING 
print("\n🔍 Tuning XGBoost...")
xgb_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [1, 1.5, 2]
}
xgb_search = RandomizedSearchCV(
    XGBRegressor(random_state=42, n_jobs=-1),
    xgb_params, n_iter=15, scoring='r2', cv=3, n_jobs=-1, random_state=42
)
xgb_search.fit(X_train, y_train)
best_xgb = xgb_search.best_estimator_
print(f"Best XGB Params: {xgb_search.best_params_}")
results['XGBoost'] = {'model': best_xgb, 'preds': best_xgb.predict(X_test)}

#  EVALUATION METRICS
metrics = []
for name, result in results.items():
    preds_train = result['model'].predict(X_train)
    preds_test = result['preds']
    metrics.append([
        name,
        r2_score(y_train, preds_train),
        r2_score(y_test, preds_test),
        mean_absolute_error(y_test, preds_test),
        np.sqrt(mean_squared_error(y_test, preds_test))
    ])
df_metrics = pd.DataFrame(metrics, columns=['Model', 'Train R²', 'Test R²', 'MAE', 'RMSE'])
print("\nModel Performance:")
print(df_metrics)

# VISUAL COMPARISON 
plt.figure(figsize=(10, 6))
df_melt = df_metrics.melt(id_vars='Model', var_name='Metric', value_name='Value')
sns.barplot(data=df_melt, x='Model', y='Value', hue='Metric', palette='Set2')
plt.title("Model Comparison Metrics")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# SHAP IMPORTANCE & MODEL SELECTION
def calculate_score(row):
    base_score = (row['Test R²'] * 100) - (row['RMSE'] * 2) - (row['MAE'] * 1.5)
    return base_score

df_metrics['score'] = df_metrics.apply(calculate_score, axis=1)

# Force preference for XGBoost if within a tolerance
xgb_row = df_metrics[df_metrics['Model'] == 'XGBoost'].iloc[0]
best_overall_row = df_metrics.iloc[df_metrics['score'].idxmax()]

tolerance = 0.03  # 3% R² drop allowed
if (best_overall_row['Test R²'] - xgb_row['Test R²']) <= tolerance:
    best_model_name = 'XGBoost'
else:
    best_model_name = best_overall_row['Model']

print(f"\nComputing SHAP for chosen model: {best_model_name}")
best_model = results[best_model_name]['model']
explainer = shap.Explainer(best_model, X_train)
shap_values = explainer(X_test)
shap.plots.beeswarm(shap_values, max_display=10)

# Save chosen model
model_filename = f"saved_models/best_model_{best_model_name.lower().replace(' ', '_')}.pkl"
joblib.dump(best_model, model_filename)
print(f"Best model saved: {model_filename}")
