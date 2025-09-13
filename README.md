# Explainable AI for Credit Risk Assessment

## Project Overview
This project predicts the risk of loan default for credit applicants using machine learning and delivers transparent explanations for each prediction. The goal is to enhance trust and decision-making in consumer lending by combining predictive accuracy with model interpretability.

## Dataset
- **Source:** [Give Me Some Credit - Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit)  
- **Rows/Columns:** 150,000 rows, 12 columns  
- **Features:** Age, Monthly Income, Debt Ratio, Credit Utilization, Payment History, Number of Dependents, etc.  
- **Target Variable:** `SeriousDlqin2yrs` (1 = default, 0 = no default)  
- **Imbalance:** Only ~7% of customers are high risk  

---

## Data Exploration
- Explored data types, distributions, and missing values.  
- Substantial missing data noted in **Monthly Income** and **Number of Dependents**.  
- **Key insight:** Payment history and debt ratio are closely tied to default risk.  

---

## Data Preprocessing
- Removed rows with missing values for simplicity (final size: 120,269 rows).  
- Scaled all numeric features to handle outliers and varying units.  
- Addressed class imbalance using strategies such as class weighting or SMOTE.  

---

## Modeling
- Split data into **train/test sets** (e.g., 80/20).  
- Trained models: Logistic Regression, Random Forest, XGBoost.  
- Evaluated with **precision, recall, F1-score, ROC-AUC** due to class imbalance.  
- **Best model:** [Your best model, e.g., XGBoost], ROC-AUC: [your score].  

---

## Explainability
- Used **SHAP** to explain predictions globally and locally.  
- Feature contributions visualized for each prediction.  
- **Example:** Probability of default = 18.93% (Low Risk); main factors decreasing risk were age, low debt ratio, no major late payments.  

---

## Application
- Built a **Flask web app** for real-time prediction and explanation.  
- Users input applicant details and receive:  
  - Risk level  
  - Probability of default  
  - Feature explanations  
- Transparent and easy to understand for both technical and non-technical users.  

---

## How to Run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Navigate to the app folder and run the Flask app
cd app
python app.py

# 3. Access the application
# Open your browser and go to:
http://127.0.0.1:5000/

```
## Key Findings
- Payment history and debt ratio are critical predictors of credit risk.
- SHAP-based explanations increase trust in AI-driven credit decisions.
- Transparent modeling helps with **regulatory compliance** and **business adoption**.


## References
1. Explainable Artificial Intelligence Credit Risk Assessment using Machine Learning
2. Artificial Intelligence and Machine Learning in Credit Risk Assessment
3. Kaggle “Give Me Some Credit” dataset
