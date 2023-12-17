from flask import Flask, request, jsonify 
import numpy as np
import pickle

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello Patient"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        # gender	age	hypertension	heart_disease	ever_married	work_type	Residence_type	avg_glucose_level	bmi	smoking_status	stroke
        gender = float(request.form.get('gender'))
        age = float(request.form.get('age'))
        hypertension = float(request.form.get('hypertension'))
        heart_disease = float(request.form.get('heart_disease'))
        ever_married = float(request.form.get('ever_married'))
        work_type = float(request.form.get('work_type'))
        Residence_type = float(request.form.get('Residence_type'))
        glucose_level = float(request.form.get('avg_glucose_level'))
        bmi = float(request.form.get('bmi'))
        smoking = float(request.form.get('smoking_status'))

        input_query = np.array([[gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, glucose_level, bmi, smoking]])
        
        result = model.predict(input_query)[0]

        return jsonify({'stroke': str(result)})
    except Exception as e:
        return jsonify({'error': str(e)})
    
if __name__ == '__main__':
    app.run(debug = True)