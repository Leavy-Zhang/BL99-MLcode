import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

base_dir = os.path.abspath('.')
output_dir = os.path.join(base_dir, 'ML', 'finalTry')

results_df = pd.read_csv(os.path.join(output_dir, 'model_validation_results.csv'))
lasso_coefs = pd.read_csv(os.path.join(output_dir, 'lasso_selected_features.csv'))
elastic_coefs = pd.read_csv(os.path.join(output_dir, 'elasticnet_selected_features.csv'))
lr_en_coefs = pd.read_csv(os.path.join(output_dir, 'logistic_elasticnet_selected_features.csv'))

fig, ax = plt.subplots(figsize=(10, 6))
metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
bar_width = 0.25
positions = np.arange(len(metrics))

for i, model_name in enumerate(results_df['model']):
    ax.bar(positions + i * bar_width, results_df[results_df['model'] == model_name][metrics].values[0], 
           width=bar_width, label=model_name)

ax.set_xticks(positions + bar_width)
ax.set_xticklabels(metrics)
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'model_performance.pdf'))
plt.close(fig)
print("Created: model_performance.pdf")

top_15_lasso = lasso_coefs.head(15)
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x='abs_coefficient', y='metabolite', data=top_15_lasso, palette='viridis')
ax.set_xlabel('Absolute Coefficient')
ax.set_ylabel('Metabolite')
ax.set_title('Top 15 Metabolites - LASSO')
plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'lasso_features.pdf'))
plt.close(fig)
print("Created: lasso_features.pdf")

top_15_elastic = elastic_coefs.head(15)
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x='abs_coefficient', y='metabolite', data=top_15_elastic, palette='viridis')
ax.set_xlabel('Absolute Coefficient')
ax.set_ylabel('Metabolite')
ax.set_title('Top 15 Metabolites - ElasticNet')
plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'elasticnet_features.pdf'))
plt.close(fig)
print("Created: elasticnet_features.pdf")

top_15_lr_en = lr_en_coefs.head(15)
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x='abs_coefficient', y='metabolite', data=top_15_lr_en, palette='viridis')
ax.set_xlabel('Absolute Coefficient')
ax.set_ylabel('Metabolite')
ax.set_title('Top 15 Metabolites - Logistic ElasticNet')
plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'logistic_elasticnet_features.pdf'))
plt.close(fig)
print("Created: logistic_elasticnet_features.pdf")

comparison_df = pd.read_csv(os.path.join(output_dir, 'feature_selection_comparison.csv'))
selected_counts = comparison_df['selected_by'].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(8, 5))
selected_counts.plot(kind='bar', ax=ax, color=['#4CAF50', '#2196F3', '#FF9800'])
ax.set_xlabel('Number of Models Selected')
ax.set_ylabel('Number of Metabolites')
ax.set_title('Feature Selection Consistency')
plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'feature_consistency.pdf'))
plt.close(fig)
print("Created: feature_consistency.pdf")

fig, ax = plt.subplots(figsize=(8, 4))
sns.countplot(x='selected_by', data=comparison_df, palette='Set2')
ax.set_xlabel('Number of Models')
ax.set_ylabel('Count')
ax.set_title('Distribution of Feature Selection')
plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'feature_distribution.pdf'))
plt.close(fig)
print("Created: feature_distribution.pdf")

print("\nAll PDF plots generated successfully!")