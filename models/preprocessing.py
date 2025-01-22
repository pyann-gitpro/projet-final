import speech_recognition as sr
from nltk.stem import WordNetLemmatizer
import re


def preprocessing(dataset, num_of_rows=1):
    stemmer = WordNetLemmatizer()  # Initialisation du lemmatiseur pour la réduction des mots à leur forme de base
    corpus = []  # Liste pour stocker les textes nettoyés

    for i in range(0, num_of_rows):
        # Suppression des caractères spéciaux
        document = re.sub(r'\W', ' ', dataset[i])

        # Suppression des caractères isolés dans le texte
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Suppression des caractères isolés en début de ligne
        document = re.sub(r'^\s*[a-zA-Z]\s+', ' ', document)

        # Remplacement des espaces multiples par un seul espace
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Conversion du texte en minuscules
        document = document.lower()

        # Séparation des mots
        document = document.split()

        # Lemmatisation des mots pour les réduire à leur forme de base
        document = [stemmer.lemmatize(word) for word in document]

        # Reconstruction du texte prétraité
        document = ' '.join(document)

        # Ajout du texte nettoyé au corpus
        corpus.append(document)

    return corpus