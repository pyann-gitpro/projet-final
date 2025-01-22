import streamlit as st
import requests

# URL de l'API FastAPI
API_URL = "http://127.0.0.1:8000/predict"

# Titre de l'application
st.title("Prédiction des Appels Frauduleux")

# Champ pour entrer le contenu de l'appel
call_content = st.text_area("Entrez le contenu de l'appel :", "")

# Bouton pour envoyer la requête
if st.button("Prédire"):
    if call_content.strip():
        try:
            # Envoyer la requête à l'API
            response = requests.post(
                API_URL,
                json={"call_content": call_content.strip()}
            )
            
            # Vérifier si la requête a réussi
            if response.status_code == 200:
                prediction = response.json()["prediction"]
                st.success(f"La prédiction est : {prediction}")
            else:
                st.error(f"Erreur API : {response.status_code} - {response.json().get('detail', 'Erreur inconnue')}")
        except Exception as e:
            st.error(f"Une erreur est survenue : {str(e)}")
    else:
        st.warning("Veuillez entrer un contenu pour effectuer la prédiction.")
