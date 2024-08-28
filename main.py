from typing import Optional
from fastapi import FastAPI


app = FastAPI()

@app.get('/')

def root():
    return {"message": "We ARE APIERS"}


@app.get("/items/{item_id}")
async def read_item(item_id):
    return {"item_id": item_id}


import joblib 

model = joblib.load("Log_Reg.joblib")
scaler = joblib.load("Scaler.joblib")

from pydantic import BaseModel

from pydantic import BaseModel

class InputFeatures(BaseModel):
        Mileage: int
        Engine_Size: float
    

def preprocessing(input_features: InputFeatures):
    dict_f = {
        'Mileage': input_features.Mileage,
        'Engine_Size': input_features.Engine_Size
    
    }
    feature_list = [dict_f[key] for key in sorted(dict_f)]
    return scaler.transform([list(dict_f.values())])
    

@app.get("/predict")
def predict(input_features: InputFeatures):
    return preprocessing(input_features)

@app.post("/predict")

async def predict(input_features: InputFeatures):
    data = preprocessing(input_features)
    y_pred = model.predict(data)
    return {'pred': y_pred.tolist()[0]}

