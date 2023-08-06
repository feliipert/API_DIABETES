

from builtins import dict
import pickle
import pandas as pd
from fastapi import FastAPI

# Cargamos el modelo previamente entrenado
with open('models/modelo_RF.pkl', 'rb') as gb:  # rb lectura
    modelo = pickle.load(gb)

app = FastAPI()

@app.get('/')
def hello():
    return {'message': 'Hello World'}

@app.post('/predict')
def predict(request: dict):
    # Get the data from the POST request.
    data = request['data']

    # Convert the data into a DataFrame with appropriate column names
    feature_names = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level',
                     'blood_glucose_level']
    data_df = pd.DataFrame(data, columns=feature_names)

    # Make prediction using modelo loaded from disk as per the data
    prediction = modelo.predict(data_df)
    output = prediction[0]

    return {'prediction': int(output)}  # Convert the output to int to ensure JSON serialization

