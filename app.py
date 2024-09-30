from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved machine learning model
model = joblib.load('model.pkl')

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        # Get the input values from the form
        Gender = int(request.form['Gender'])
        Married = int(request.form['Married'])
        Dependents = float(request.form['Dependents'])
        Education = int(request.form['Education'])
        Self_Employed = int(request.form['Self_Employed'])
        ApplicantIncome = float(request.form['ApplicantIncome'])
        CoapplicantIncome = float(request.form['CoapplicantIncome'])
        LoanAmount = float(request.form['LoanAmount'])
        Loan_Amount_Term = float(request.form['Loan_Amount_Term'])
        Credit_History = float(request.form['Credit_History'])
        Property_Area = int(request.form['Property_Area'])

        # Create a numpy array with the input values
        input_values = np.array([[Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area]])

        # Use the machine learning model to make a prediction
        prediction = model.predict(input_values)

        # Get the predicted class label (0 or 1)
        predicted_class = prediction[0]

        # Map the predicted class label to a human-readable result
        if predicted_class == 0:
            result = 'Loan Not Approved'
        else:
            result = 'Loan Approved'

        # Render the result page with the predicted result
        return render_template('result.html', result=result)

    # Render the main page with the form
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)