# COS40007_Week2_Assignment2

# 📊 Portfolio Report – Assignment 1

**Name:** [Your Name]  
**Student Number:** [Your Student Number]  
**Studio Class:** [Your Studio]  

---

## Executive Summary
This report presents the analysis of the **Breast Cancer Wisconsin dataset** for binary classification (malignant vs benign).  
Using feature engineering, feature selection, and a Decision Tree model, we evaluated the impact of different feature sets on model performance.  
Results show that combining selected original features with engineered features provides the best predictive accuracy.

---

## 1. Dataset Overview
- **Dataset:** Breast Cancer Wisconsin (CSV, 569 samples × 30 features)  
- **Target Variable:** Diagnosis (`M = malignant = 0`, `B = benign = 1`)  
- **Class Distribution:** Benign = 357, Malignant = 212  
- **Why this dataset?** Widely used in medical ML tasks, good for classification and interpretability.  

---

## 2. Data Preparation
### 2.1 Class Labelling
- Diagnosis mapped as: `M → 0`, `B → 1`

### 2.2 Normalisation
- All 30 numeric features scaled to [0, 1] using **Min–Max scaling**

### 2.3 Feature Engineering
Created 5 new features:
1. `area_to_radius = area1 / radius1`
2. `perimeter_to_radius = perimeter1 / radius1`
3. `symmetry_over_smoothness = symmetry1 / smoothness1`
4. `concavity_times_area = concavity1 × area1`
5. `texture_squared = (texture1)²`

---

## 3. Feature Sets Evaluated
1. **All_original (30)** – all original features  
2. **Mean_features (10)** – features ending in `1` (mean values)  
3. **Top10_corr (10)** – top 10 most correlated features  
4. **Engineered (5)** – engineered features only  
5. **Selected_combination (10)** – top 5 correlated + 5 engineered  

---

## 4. Training and Decision Tree Model Development
- **Algorithm:** Decision Tree Classifier (`random_state=42`)  
- **Validation:** Stratified 5-fold cross-validation  
- **Metrics:** Accuracy, Precision (macro), Recall (macro), F1 (macro)  

---

## 5. Final Comparison Table

| Feature Set             | n_features | Accuracy | Precision | Recall | F1   |
|--------------------------|-----------:|----------|-----------|--------|------|
| All_original             | 30         | 0.903    | 0.902     | 0.902  | 0.903 |
| Mean_features            | 10         | 0.909    | 0.906     | 0.906  | 0.906 |
| Top10_corr               | 10         | 0.894    | 0.893     | 0.894  | 0.894 |
| Engineered               | 5          | 0.898    | 0.898     | 0.898  | 0.898 |
| **Selected_combination** | 10         | **0.925** | **0.926** | **0.925** | **0.925** |

---

## 6. Summary of Observations
- **Selected_combination** performed best (~92.5% accuracy), confirming the benefit of combining engineered + original features.  
- **Mean_features** were nearly as strong as using all 30 features, showing efficiency.  
- **Engineered features** alone gave ~89.8%, proving they add predictive value.  
- **Top10_corr** underperformed, showing correlation alone isn’t optimal.  

**Key Insight:** Domain-driven engineered features, when combined with selected predictors, yield stronger models than raw features alone.

---

## 7. Conclusion
- A hybrid feature set improved accuracy by ~2% compared to using all features.  
- **Recommendation:** Extend analysis with hyperparameter tuning, Random Forests, Gradient Boosting, and ROC-AUC evaluation.  

---

## 8. Appendix
- Dataset: `breast_cancer.csv`  
- Code: `ass1.py`  
- Outputs: results & comparison table  
---
