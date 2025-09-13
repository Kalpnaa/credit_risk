from flask import Flask, render_template, request
from joblib import load
import numpy as np
import shap
import os

app = Flask(__name__)

# Load model
model = load('models/credit_risk_model.joblib')

# Initialize SHAP explainer once globally
explainer = shap.TreeExplainer(model)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input fields
        age = float(request.form['applicant_age'])
        monthly_income = float(request.form['monthly_income'])
        debt_ratio = float(request.form['debt_ratio'])
        open_credit_lines = float(request.form['open_credit_lines'])
        dependents = float(request.form['dependents'])
        credit_utilization = float(request.form['credit_utilization'])
        late_30_59 = float(request.form['late_30_59'])
        late_90 = float(request.form['late_90'])
        real_estate_loans = float(request.form['real_estate_loans'])
        late_60_89 = float(request.form['late_60_89'])


        # Build feature array in correct order
        features = np.array([[
            credit_utilization,
            age,
            late_30_59,
            debt_ratio,
            monthly_income,
            open_credit_lines,
            late_90,
            dependents,
            real_estate_loans,
            late_60_89
        ]])

        # Apply preprocessing if needed (e.g., scaler.transform)

        # Make prediction
        pred_prob = model.predict_proba(features)[0][1]
        pred_label = "High Risk" if pred_prob > 0.5 else "Low Risk"

        # SHAP explanation
        shap_values = explainer.shap_values(features)

         # Ensure shap_values is a numpy array of shape (1, 10)
        if isinstance(shap_values, list):
           shap_values = shap_values[0]

        contributions = dict(zip(
        ['RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse',
        'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
        'NumberOfTimes90DaysLate', 'NumberOfDependents',
        'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse'],
         shap_values[0]
      ))


        sorted_contrib = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)

        return render_template('result.html',
                               prediction=pred_label,
                               probability=round(pred_prob * 100, 2),
                               contributions=sorted_contrib)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
