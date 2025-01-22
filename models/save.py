import joblib
def save_model(best_model, filename='./models/name_model.pkl'):
        """
    Sauvegarde le modèle entraîné dans un fichier .pkl à l'emplacement
    ./models/nale_model.pkl

    Parameters:
        best_model_knn: le modèle entraîné

    Returns:
        best_model_knn: le modèle sauvegardé
    """
        saved_model = joblib.dump(best_model, filename)
        return saved_model