import pandas as pd
import numpy as np
import os

output_dir = '../output/'
os.makedirs(output_dir, exist_ok=True)

delta_df = pd.read_csv('../inputData/delta_combined.csv', encoding='latin-1')
groups_df = pd.read_csv('../inputData/samples.txt', sep='\t', encoding='latin-1')
merged_df = pd.merge(delta_df, groups_df, on='donor')

remove_samples = ['P57', 'P55', 'P76', 'P2']
filtered_df = merged_df[~merged_df['donor'].isin(remove_samples)].copy()
filtered_df['class'] = filtered_df['group'].map({'A': 0, 'B': 1})

feature_cols = [col for col in filtered_df.columns if col not in ['donor', 'mode', 'group', 'response', 'class']]
X = filtered_df[feature_cols].values
y = filtered_df['class'].values

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

from sklearn.linear_model import Lasso, ElasticNet, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

lasso_cv = GridSearchCV(Lasso(max_iter=10000), {'alpha': np.logspace(-6, 2, 30)}, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
lasso_cv.fit(X_train_scaled, y_train)
best_lasso = lasso_cv.best_estimator_

elastic_cv = GridSearchCV(ElasticNet(max_iter=10000), {'alpha': np.logspace(-6, 2, 20), 'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]}, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
elastic_cv.fit(X_train_scaled, y_train)
best_elastic = elastic_cv.best_estimator_

lr_en_cv = GridSearchCV(LogisticRegression(max_iter=10000), {'C': np.logspace(-4, 4, 30), 'penalty': ['elasticnet'], 'solver': ['saga'], 'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]}, cv=5, scoring='roc_auc', n_jobs=-1)
lr_en_cv.fit(X_train_scaled, y_train)
best_lr_en = lr_en_cv.best_estimator_

models = {'LASSO': best_lasso, 'ElasticNet': best_elastic, 'Logistic_ElasticNet': best_lr_en}
results_list = []

for model_name, model in models.items():
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
        y_pred = model.predict(X_val_scaled)
    else:
        y_pred = (model.predict(X_val_scaled) >= 0.5).astype(int)
        y_pred_proba = model.predict(X_val_scaled)
    
    results_list.append({
        'model': model_name,
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, zero_division=0),
        'recall': recall_score(y_val, y_pred, zero_division=0),
        'f1': f1_score(y_val, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_val, y_pred_proba)
    })

results_df = pd.DataFrame(results_list)
results_df.to_csv(f'{output_dir}/model_validation_results.csv', index=False)

lasso_coefs = pd.DataFrame({'metabolite': feature_cols, 'coefficient': best_lasso.coef_, 'abs_coefficient': np.abs(best_lasso.coef_)}).sort_values('abs_coefficient', ascending=False)
lasso_coefs.to_csv(f'{output_dir}/lasso_coefficients.csv', index=False)
lasso_coefs[lasso_coefs['abs_coefficient'] > 0].to_csv(f'{output_dir}/lasso_selected_features.csv', index=False)

elastic_coefs = pd.DataFrame({'metabolite': feature_cols, 'coefficient': best_elastic.coef_, 'abs_coefficient': np.abs(best_elastic.coef_)}).sort_values('abs_coefficient', ascending=False)
elastic_coefs.to_csv(f'{output_dir}/elasticnet_coefficients.csv', index=False)
elastic_coefs[elastic_coefs['abs_coefficient'] > 0].to_csv(f'{output_dir}/elasticnet_selected_features.csv', index=False)

lr_en_coefs = pd.DataFrame({'metabolite': feature_cols, 'coefficient': best_lr_en.coef_[0], 'abs_coefficient': np.abs(best_lr_en.coef_[0])}).sort_values('abs_coefficient', ascending=False)
lr_en_coefs.to_csv(f'{output_dir}/logistic_elasticnet_coefficients.csv', index=False)
lr_en_coefs[lr_en_coefs['abs_coefficient'] > 0].to_csv(f'{output_dir}/logistic_elasticnet_selected_features.csv', index=False)

lasso_features = set(lasso_coefs[lasso_coefs['abs_coefficient'] > 0]['metabolite'].tolist())
elastic_features = set(elastic_coefs[elastic_coefs['abs_coefficient'] > 0]['metabolite'].tolist())
lr_en_features = set(lr_en_coefs[lr_en_coefs['abs_coefficient'] > 0]['metabolite'].tolist())

comparison_df = pd.DataFrame({
    'metabolite': list(lasso_features | elastic_features | lr_en_features),
    'LASSO': [m in lasso_features for m in lasso_features | elastic_features | lr_en_features],
    'ElasticNet': [m in elastic_features for m in lasso_features | elastic_features | lr_en_features],
    'Logistic_ElasticNet': [m in lr_en_features for m in lasso_features | elastic_features | lr_en_features]
})
comparison_df['selected_by'] = comparison_df[['LASSO', 'ElasticNet', 'Logistic_ElasticNet']].sum(axis=1)
comparison_df = comparison_df.sort_values('selected_by', ascending=False)
comparison_df.to_csv(f'{output_dir}/feature_selection_comparison.csv', index=False)

with open(f'{output_dir}/analysis_summary.txt', 'w', encoding='utf-8') as f:
    f.write("Final Analysis Report\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"A group: {sum(y == 0)} samples\n")
    f.write(f"B group: {sum(y == 1)} samples\n")
    f.write(f"Total metabolites: {len(feature_cols)}\n\n")
    f.write("Models: LASSO, ElasticNet, Logistic_ElasticNet\n\n")
    f.write("Validation Results:\n")
    f.write("-" * 50 + "\n")
    for _, row in results_df.iterrows():
        f.write(f"\n{row['model']}:\n")
        f.write(f"  Accuracy: {row['accuracy']:.4f}\n")
        f.write(f"  Precision: {row['precision']:.4f}\n")
        f.write(f"  Recall: {row['recall']:.4f}\n")
        f.write(f"  F1: {row['f1']:.4f}\n")
        f.write(f"  ROC AUC: {row['roc_auc']:.4f}\n")
    f.write(f"\nLASSO selected: {len(lasso_features)} metabolites\n")
    f.write(f"ElasticNet selected: {len(elastic_features)} metabolites\n")
    f.write(f"Logistic_ElasticNet selected: {len(lr_en_features)} metabolites\n")
    f.write(f"Common across all: {len(lasso_features & elastic_features & lr_en_features)} metabolites\n")

print("Done")
