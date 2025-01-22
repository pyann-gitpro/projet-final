from typing import Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer

import sqlite3
import pandas as pd
import joblib
import os

# Chemins des fichiers
bd_path = '../bdd/fraud_calls.db'
model_path = '../module/best_random_forest_model.pkl'

# Vérifier si le modèle existe
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Le fichier modèle '{model_path}' est introuvable.")

# Charger le modèle
model_rf = joblib.load(model_path)

# Définir l'application FastAPI
app = FastAPI()

# Endpoint pour récupérer les données de la base
@app.get("/data/")
def get_data(limit: int = 5):
    if not os.path.exists(bd_path):
        raise HTTPException(status_code=404, detail="La base de données est introuvable.")
    
    conn = sqlite3.connect(bd_path)
    try:
        query = f"SELECT * FROM fraud_calls LIMIT {limit}"
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des données : {str(e)}")
    finally:
        conn.close()
    
    return df.to_dict(orient="records")

# Définir la structure de la requête pour la prédiction
class CallRequest(BaseModel):
    call_content: str

tfidf = TfidfVectorizer()


# Endpoint pour effectuer une prédiction
@app.post("/predict")
def predict(call: CallRequest):
    try:
        # Prétraiter le texte
        transformed_text = tfidf.transform([call.call_content])
        # Faire la prédiction
        prediction = model_rf.predict(transformed_text)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")
    
    return {"prediction": prediction}

# @app.post("/predict")
# def predict(call: CallRequest):
#     try:
#         prediction = model_rf.predict([call.call_content])[0]
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")
    
#     return {"prediction": prediction}

