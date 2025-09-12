import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

# 1. Load CSV
df = pd.read_csv("breast_cancer.csv")
print(df.head())
print(df.info())
print(df.describe())
# Target column
y = df['Diagnosis'].map({'M': 0, 'B': 1}).rename("target")
X = df.drop(columns=['Diagnosis'])


print("Shape:", X.shape)
print("Classes:\n", y.value_counts())

# 2. Normalisation
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 3. Feature engineering (using the "1" group as 'mean' values)
eng = pd.DataFrame(index=X.index)
eng['area_to_radius'] = X['area1'] / X['radius1']
eng['perimeter_to_radius'] = X['perimeter1'] / X['radius1']
eng['symmetry_over_smoothness'] = X['symmetry1'] / X['smoothness1']
eng['concavity_times_area'] = X['concavity1'] * X['area1']
eng['texture_squared'] = X['texture1'] ** 2
eng.fillna(0, inplace=True)

eng_scaled = pd.DataFrame(MinMaxScaler().fit_transform(eng), columns=eng.columns)

# Final combined dataset
final_scaled = pd.concat([X_scaled, eng_scaled], axis=1)
final_scaled['target'] = y

# 4. Feature sets
feat_all = list(X.columns)
feat_mean = [c for c in X.columns if c.endswith("1")]  # "1" = mean group

tmp_corr = pd.concat([X, eng, y], axis=1).corr()['target'].abs().drop('target')
top10 = list(tmp_corr.sort_values(ascending=False).head(10).index)

feat_eng = list(eng.columns)
top5_orig = [f for f in top10 if f in X.columns][:5]
feat_selected = top5_orig + feat_eng

feature_sets = {
    'All_original': feat_all,
    'Mean_features': feat_mean,
    'Top10_corr': top10,
    'Engineered': feat_eng,
    'Selected_combination': feat_selected
}

# 5. Train & evaluate Decision Tree
clf = DecisionTreeClassifier(random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

results = []
for name, feats in feature_sets.items():
    Xf = final_scaled[feats].values
    scores = cross_validate(clf, Xf, y.values, cv=cv, scoring=scoring)
    results.append({
        'feature_set': name,
        'n_features': len(feats),
        'accuracy': scores['test_accuracy'].mean(),
        'precision': scores['test_precision_macro'].mean(),
        'recall': scores['test_recall_macro'].mean(),
        'f1': scores['test_f1_macro'].mean()
    })

results_df = pd.DataFrame(results).set_index('feature_set')
print("\nComparison of feature sets:\n", results_df)
