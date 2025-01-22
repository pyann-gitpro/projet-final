import sqlite3
import pandas as pd

# Connexion à la base de données (ou création de la base de données si elle n'existe pas)
conn = sqlite3.connect('./bdd/fraud_calls.db')

# Création d'un curseur pour exécuter des commandes SQL
cursor = conn.cursor()

# Création d'une table
cursor.execute('''
CREATE TABLE IF NOT EXISTS fraud_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_content TEXT NOT NULL,
    label TEXT NOT NULL
)
''')

# Lire le fichier CSV
# Remplacez 'data.csv' par le chemin de votre fichier CSV
df = pd.read_csv('./data/raw/data_fraud_calls.csv')

# Afficher les données lues pour vérification
print(df.head())

# Insérer les données dans la table
df.to_sql('fraud_calls', conn, if_exists='append', index=False)

# Valider les changements
conn.commit()

# Fermer la connexion
conn.close()