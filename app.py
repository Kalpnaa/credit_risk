from flask import Flask, render_template, request
from joblib import load
import numpy as np
import shap

app = Flask(__name__)

# Load model
model = load('models/credit_risk_model.joblib')

# Initialize SHAP explainer once globally
explainer = shap.TreeExplainer(model)

def scale_features(features):
    """
    Dynamically scales input features using standard scaling (mean=0, std=1)
    Input:
        features: numpy array of shape (1, n_features)
    Output:
        scaled_features: numpy array of same shape
    """
    features = np.array(features, dtype=float)
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    std[std == 0] = 1  # prevent division by zero
    return (features - mean) / std

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input fields from form
        input_data = {
            'RevolvingUtilizationOfUnsecuredLines': float(request.form['credit_utilization']),
            'age': float(request.form['applicant_age']),
            'NumberOfTime30-59DaysPastDueNotWorse': float(request.form['late_30_59']),
            'DebtRatio': float(request.form['debt_ratio']),
            'MonthlyIncome': float(request.form['monthly_income']),
            'NumberOfOpenCreditLinesAndLoans': float(request.form['open_credit_lines']),
            'NumberOfTimes90DaysLate': float(request.form['late_90']),
            'NumberOfDependents': float(request.form['dependents']),
            'NumberRealEstateLoansOrLines': float(request.form['real_estate_loans']),
            'NumberOfTime60-89DaysPastDueNotWorse': float(request.form['late_60_89'])
        }

        # Build feature array in correct order
        feature_order = [
            'RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse',
            'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
            'NumberOfTimes90DaysLate', 'NumberOfDependents',
            'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse'
        ]
        features = np.array([[input_data[feat] for feat in feature_order]])

        # Scale features
        features_scaled = scale_features(features)

        # Make prediction
        pred_prob = model.predict_proba(features_scaled)[0][1]
        pred_label = "High Risk" if pred_prob > 0.5 else "Low Risk"

        # SHAP explanation
        shap_values = explainer.shap_values(features_scaled)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        contributions = dict(zip(feature_order, shap_values[0]))
        sorted_contrib = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)

        return render_template('result.html',
                               prediction=pred_label,
                               probability=round(pred_prob * 100, 2),
                               contributions=sorted_contrib)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
