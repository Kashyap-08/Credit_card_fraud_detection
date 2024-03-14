from src.CreditCardFraudDetection.pipelines.prediction_pipeline import CustomData, Predict_Pipeline
from src.CreditCardFraudDetection.logger import logging

from flask import Flask, request, render_template, jsonify
from datetime import datetime

app = Flask(__name__)

# @app.route('/predict')
# def home_page():
#     return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('form.html')
    
    else:
        custom_date = CustomData(
            Time= request.form.get('Time'),
            V1=request.form.get('V1'),
            V2=request.form.get('V2'),
            V3=request.form.get('V3'),
            V4=request.form.get('V4'),
            V5=request.form.get('V5'),
            V6=request.form.get('V6'),
            V7=request.form.get('V7'),
            V8=request.form.get('V8'),
            V9=request.form.get('V9'),
            V10=request.form.get('V10'),
            V11=request.form.get('V11'),
            V12=request.form.get('V12'),
            V13=request.form.get('V13'),
            V14=request.form.get('V14'),
            V15=request.form.get('V15'),
            V16=request.form.get('V16'),
            V17=request.form.get('V17'),
            V18=request.form.get('V18'),
            V19=request.form.get('V19'),
            V20=request.form.get('V20'),
            V21=request.form.get('V21'),
            V22=request.form.get('V22'),
            V23=request.form.get('V23'),
            V24=request.form.get('V24'),
            V25=request.form.get('V25'),
            V26=request.form.get('V26'),
            V27=request.form.get('V27'),
            V28=request.form.get('V28'),
            Amount=request.form.get("Amount")
        )

        df = custom_date.get_data_as_dataframe()

        predict_obj = Predict_Pipeline()
        predicted_value = predict_obj.predict(df)
        predicted_value = predicted_value[0]
        print("Predicted Value is:", predicted_value)
        val = 'The Transaction is Legitimate :)'
        if predicted_value:
            val = 'The Transaction is Fraudulent :o'
        
        return render_template('form.html', final_result = val)

if __name__ == "__main__":
    app.run(host = "0.0.0.0", port=8080,debug=True)