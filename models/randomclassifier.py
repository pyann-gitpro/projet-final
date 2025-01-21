import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from preprocessing import preprocessing
from load_dataset import load_data
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report

def predict_using_random_forest(dataset, num_of_rows, callData):
    # Prétraitement des données textuelles
    x = preprocessing(dataset['call_content'], num_of_rows)
    y = dataset.Label

    # Division des données en ensembles d'entraînement (70%) et de test (30%)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Vectorisation des textes
    count_vectorizer = CountVectorizer(max_features=3000, min_df=3, max_df=0.8, stop_words=stopwords.words('english'), ngram_range=(1, 2))
    count_train = count_vectorizer.fit_transform(x_train)
    count_test = count_vectorizer.transform(x_test)

    # Définir le modèle Random Forest
    model = RandomForestClassifier(random_state=42)

    # Définir la grille d'hyperparamètres à tester
    param_grid = {
        'n_estimators': [50, 100],  # Nombre d'arbres dans la forêt
        'max_depth': [None, 10, 20],  # Profondeur maximale des arbres
        'min_samples_split': [2, 5],  # Nombre minimum d'échantillons requis pour diviser un nœud
        'min_samples_leaf': [1, 2]  # Nombre minimum d'échantillons requis pour être à une feuille
    }

    # Configurer GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1)

    # Entraîner le modèle avec GridSearchCV
    grid_search.fit(count_train, y_train)

    # Meilleurs hyperparamètres
    print("Meilleurs hyperparamètres : ", grid_search.best_params_)

    # Prédictions avec le meilleur modèle
    y_pred = grid_search.predict(count_test)

    # Évaluation
    print(classification_report(y_test, y_pred, target_names=['normal', 'fraud']))

    # Sauvegarder le meilleur modèle
    joblib.dump(grid_search.best_estimator_, './module/best_random_forest_model.pkl')  # Sauvegarde du meilleur modèle
    joblib.dump(count_vectorizer, './module/count_vectorizer_rf.pkl')  # Sauvegarde du vectoriseur

    return callData, y_pred

# Charger les données
dataset = load_data()

# Définir le nombre de lignes à utiliser pour le prétraitement
num_of_rows = len(dataset)
callData = "Todays Vodafone numbers ending with 4882 are selected to a receive a £350 award. If your number matches call 09064019014 to receive your £350 award"

# Appeler la fonction de prédiction
result = predict_using_random_forest(dataset, num_of_rows, callData)  

# Fitting 5 folds for each of 24 candidates, totalling 120 fits
# Meilleurs hyperparamètres :  {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 50}
#               precision    recall  f1-score   support

#       normal       0.98      0.83      0.90       200
#        fraud       0.98      1.00      0.99      1578

#     accuracy                           0.98      1778
#    macro avg       0.98      0.91      0.94      1778
# weighted avg       0.98      0.98      0.98      1778