# import os
# import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from preprocessing import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
#from sklearn import metrics
from load_dataset import load_data
import joblib


def predict_using_count_vectoriser(dataset, num_of_rows, callData):
    # Prétraitement des données textuelles
    x = preprocessing(dataset['call_content'], num_of_rows)
    y = dataset.Label

    # Division des données en ensembles d'entraînement (70%) et de test (30%)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Prétraitement des nouvelles données pour le test en direct
    #live_test = preprocessing(callData, len(callData))

    # Vectorisation des textes : comptage des occurrences de chaque mot avec CountVectorizer
    count_vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))

    # Ajustement du vectoriseur aux données d'entraînement et transformation en matrices
    count_train = count_vectorizer.fit_transform(x_train)
    count_test = count_vectorizer.transform(x_test)
    #live_count_test = count_vectorizer.transform(live_test)

    # Définir le modèle
    model_nb = MultinomialNB()

    # Définir la grille d'hyperparamètres à tester
    param_grid = {
        'alpha': [0.1, 0.5, 1.0, 1.5, 2.0]  # Exemple d'hyperparamètre pour MultinomialNB
    }

# Configurer GridSearchCV
    grid_search = GridSearchCV(estimator=model_nb, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1)

    # Entraîner le modèle avec GridSearchCV
    grid_search.fit(count_train, y_train)

    # Meilleurs hyperparamètres
    print("Meilleurs hyperparamètres : ", grid_search.best_params_)

    # Sauvegarder le modèle
    joblib.dump(model_nb, './module/naive_bayes_model.pkl')  # Sauvegarde du modèle
    joblib.dump(count_vectorizer, './module/count_vectorizer_nb.pkl')  # Sauvegarde du vectoriseur

    # Prédictions avec le meilleur modèle
    y_pred = grid_search.predict(count_test)

    # Évaluation
    print(classification_report(y_test, y_pred, target_names=['normal', 'fraud']))

    return callData, y_pred

# Charger les données
dataset= load_data()

# Définir le nombre de lignes à utiliser pour le prétraitement
num_of_rows = len(dataset)  # ou un nombre spécifique si vous ne voulez pas utiliser toutes les lignes

# Appeler la fonction de prédiction
callData = "Todays Vodafone numbers ending with 4882 are selected to a receive a £350 award. If your number matches call 09064019014 to receive your £350 award"
result = predict_using_count_vectoriser(dataset, num_of_rows, callData)

# Afficher les résultats
print(result)

# Fitting 5 folds for each of 5 candidates, totalling 25 fits
# Meilleurs hyperparamètres :  {'alpha': 0.1}
#               precision    recall  f1-score   support

#       normal       0.88      0.90      0.89       200
#        fraud       0.99      0.98      0.99      1578

#     accuracy                           0.98      1778
#    macro avg       0.93      0.94      0.94      1778
# weighted avg       0.98      0.98      0.98      1778

# ('Todays Vodafone numbers ending with 4882 are selected to a receive a £350 award. If your number matches call 09064019014 to receive your £350 award', array(['fraud', 'normal', 'normal', ..., 'fraud', 'fraud', 'normal'],
#       shape=(1778,), dtype='<U6'))

