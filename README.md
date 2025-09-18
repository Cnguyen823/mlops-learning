# MLOps Learning Project

- Goal: Build ML models and deploy them while learning MLOps principles.
- Skills: Python, scikit-learn, FastAPI, MLOps, AI ethics.
## 📝 Setup Progress – Week 1 (Sunday)

### Environment Setup
- [x] Created project folder: `~/mlops-learning`
- [x] Set up Python virtual environment: `venv`
- [x] Activated venv and verified Python version (`python3 --version`)
- [x] Upgraded pip inside venv and installed packages:
  - `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `fastapi`, `uvicorn`

## Day 1 - Project Initialization
- [x] Created `README.md` with project goals and plan
- [x] Initialized Git repository and made first commit
- [x] Connected local repo to GitHub and successfully pushed
- [x] Repo visible in browser: [GitHub Repository](https://github.com/Cnguyen823/mlops-learning)

## Day 2 – First ML Model

- [x] Implemented Logistic Regression on the Iris dataset.
- [x] Split data into train/test (80/20).
- [x] Achieved accuracy: ~97%.
- [x] Learned how to train, predict, and evaluate a simple classification model. See iris_model.py.

## Day 3 – Cross-Validation & Reliable Model Evaluation

- [x] Understand why single train/test splits can be misleading.
- [x] Learn about k-fold cross-validation and LOOCV.
- [x] Implement cross-validation on the Iris dataset. See iris_crossval.py.

## Day 4 – KNN, Logistic Regression, and Random Forest Comparison

- [x] Reviewed Day 3 concepts and code
- [x] Implemented K-Nearest Neighbors (KNN):
- [x] Implemented Logistic Regression:
- [x] Implemented Random Forest (optional/comparison):
- [x] Compared models using cross-validation:
- [x] Calculated accuracy, precision, recall, or RMSE
- [x] Decided which model works best for the dataset. See iris_model_comparison.py

# Day 5 — Confusion Matrices (KNN & Random Forest)
## What is a Confusion Matrix?
A confusion matrix shows **how well a classifier performs** by comparing predicted labels with actual labels.

|               | Predicted Positive | Predicted Negative |
|---------------|--------------------|--------------------|
| **Actual Positive** | True Positive (TP)   | False Negative (FN) |
| **Actual Negative** | False Positive (FP)  | True Negative (TN) |

---

## Key Metrics and Their Meaning

### 1. Accuracy
**Formula:** (TP + TN) / (TP + TN + FP + FN)  
**Meaning:** Overall correctness of the model. Tells you what fraction of predictions are right.  
⚠️ Can be misleading if the dataset is imbalanced (e.g., 95% “No” and 5% “Yes”).

---

### 2. Precision
**Formula:** TP / (TP + FP)  
**Meaning:** Out of everything the model predicted as **Positive**, how many were actually positive?  
⚠️ High precision = fewer false alarms.

---

### 3. Recall (a.k.a. Sensitivity or True Positive Rate)
**Formula:** TP / (TP + FN)  
**Meaning:** Out of all the actual positives, how many did the model correctly detect?  
⚠️ High recall = the model misses fewer positives.

---

### 4. F1 Score
**Formula:** 2 × (Precision × Recall) / (Precision + Recall)  
**Meaning:** The balance between **precision** and **recall**.  
⚠️ Useful when you want a single score and the data is imbalanced.

---

### Models Used
1. **K-Nearest Neighbors (KNN)**
   - Classification is based on the majority vote of `k` neighbors.  
   - Simple and easy to interpret.  

2. **Random Forest Classifier**
   - An ensemble of many decision trees.  
   - Each tree votes; the majority class wins.  
   - Usually more stable than a single decision tree.  

---

### Why Confusion Matrix?
- Lets you **see specific errors** instead of only accuracy.  
- Example: tells you if the model makes too many false positives (FP) vs false negatives (FN).  
- Critical in real-world tasks (medical tests, fraud detection, etc.).

- [x] Generated confusion matrices for KNN and Random Forest (confusion_matrix(y_test, y_pred))

- [x] Visualized confusion matrices using ConfusionMatrixDisplay(cm).plot() with titles and plt.show()

- [x] Implemented metrics calculation function (calculate_metrics(cm))

- [x] Calculated Accuracy, Precision, Recall, and F1 Score manually for both models

- [x] Printed metrics results for KNN and Random Forest

