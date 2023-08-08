

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
    try:
        # Get the data from the POST request.
        data = request['data']

        # Convert the data into a DataFrame with appropriate column names
        feature_names = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level',
                         'blood_glucose_level']
        data_df = pd.DataFrame(data, columns=feature_names)

        # Ensure correct data types for numeric features
        numeric_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
        data_df[numeric_features] = data_df[numeric_features].astype(float)

        # Make prediction using modelo loaded from disk as per the data
        prediction = modelo.predict(data_df)
        output = int(prediction[0])
        return {'prediction': output}
    except Exception as e:
        raise HTTPException(status_code=500, detail='Error during prediction')

    if __name__ == "__main__":
        import uvicorn
        # Especifica el puerto en el que deseas ejecutar la API (por ejemplo, 8000)
        # Cambia el parámetro 'port' según tus necesidades
        uvicorn.run(app, host="0.0.0.0", port=8000)

