from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
model = pickle.load(open('model.pkl', 'rb'))
columns = pickle.load(open('columns.pkl', 'rb'))
bin_edges = pickle.load(open('bin_edges.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html') 

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    for col in ['MonthlyCharges', 'TotalCharges', 'tenure']:
        data[col] = float(data[col])
    data['tenure_Bins'] = pd.cut([data['tenure']], bins=bin_edges, labels=[1, 2, 3, 4, 5, 6], include_lowest=True)[0]
    del data['tenure']
    df = pd.DataFrame([data])
    df['tenure_Bins'] = df['tenure_Bins'].astype(int)
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)
    prediction = model.predict(df)[0]
    result = "Yes (Customer will churn)" if prediction == 1 else "No (Customer will not churn)"
    return render_template('index.html', prediction_text=f"Prediction: {result}")

if __name__ == '__main__':
    app.run(debug=True)


