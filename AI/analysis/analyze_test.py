import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import precision_score, recall_score, f1_score

# Połączenie z bazą
engine = create_engine('sqlite:///test_results.db')

# Wczytanie tabel
tests_df = pd.read_sql_table('tests', engine)
results_df = pd.read_sql_table('test_results', engine)
defects_df = pd.read_sql_table('defects', engine)
confusion_df = pd.read_sql_table('confusion_matrices', engine)
predictions_df = pd.read_sql_table('predictions', engine)

# 1. Łączenie danych
merged_df = pd.merge(tests_df, results_df, left_on='id', right_on='test_id', how='left', suffixes=('_test', '_result'))
merged_df = pd.merge(merged_df, defects_df, left_on='id_result', right_on='test_result_id', how='left', suffixes=('', '_defect'))
merged_df = pd.merge(merged_df, confusion_df, left_on='id_result', right_on='test_result_id', how='left', suffixes=('', '_confusion'))
merged_df = pd.merge(merged_df, predictions_df, left_on='id_result', right_on='test_result_id', how='left', suffixes=('', '_prediction'))

print("Merged data preview:")
print(merged_df.head())

# 2. Podstawowe analizy
avg_accuracy_by_type = merged_df.groupby('test_type')['accuracy'].mean()
print("\nAverage accuracy by test type:")
print(avg_accuracy_by_type)

defects_per_test = merged_df.groupby('test_name')['issue_key_defect'].count()
print("\nNumber of defects per test:")
print(defects_per_test)

status_distribution = merged_df['status'].value_counts()
print("\nStatus distribution:")
print(status_distribution)

# 3. Szczegółowe analizy
if not confusion_df.empty:
    first_cm_json = confusion_df['matrix_data'].iloc[0]
    cm = np.array(json.loads(first_cm_json))
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    errors = [(i, j, cm[i, j]) for i in range(10) for j in range(10) if i != j and cm[i, j] > 0]
    top_errors = sorted(errors, key=lambda x: x[2], reverse=True)[:3]
    print("\nTop 3 misclassifications from first confusion matrix:")
    for i, j, count in top_errors:
        print(f"Class {i} misclassified as {j}: {count} times ({cm_percent[i, j]:.1f}%)")

# Per-class metrics
if not predictions_df.empty:
    y_true = np.array(json.loads(predictions_df['y_true'].iloc[0]))
    y_pred = np.array(json.loads(predictions_df['y_pred'].iloc[0]))
    per_class_precision = precision_score(y_true, y_pred, average=None)
    per_class_recall = recall_score(y_true, y_pred, average=None)
    per_class_f1 = f1_score(y_true, y_pred, average=None)
    per_class_df = pd.DataFrame({
        'Class': range(10),
        'Precision': per_class_precision,
        'Recall': per_class_recall,
        'F1-Score': per_class_f1
    })
    print("\nPer-class metrics:")
    print(per_class_df)
    per_class_df.to_csv('per_class_metrics.csv', index=False)

# Statystyki wykonania per wymaganie
stats_by_requirement = merged_df.groupby('requirement').agg({
    'accuracy': ['mean', 'min', 'max'],
    'execution_time': ['mean', 'min', 'max']
})
print("\nExecution stats by requirement:")
print(stats_by_requirement)

# 4. Wizualizacja
plt.figure(figsize=(10, 6))
sns.boxplot(x='test_type', y='accuracy', data=merged_df)
plt.title('Accuracy Distribution by Test Type')
plt.xlabel('Test Type')
plt.ylabel('Accuracy')
plt.savefig('accuracy_by_type_boxplot.png')
plt.close()

if not confusion_df.empty:
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j + 0.5, i + 0.7, f"{cm_percent[i, j]:.1f}%", ha="center", va="center", color="red", fontsize=8)
    plt.title('Confusion Matrix (First Test Result)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix_heatmap.png')
    plt.close()

# 5. Eksport do CSV
merged_df.to_csv('test_analysis_full.csv', index=False)
print("\nExported full analysis to test_analysis_full.csv")
stats_by_requirement.to_csv('stats_by_requirement.csv')
print("Exported stats by requirement to stats_by_requirement.csv")