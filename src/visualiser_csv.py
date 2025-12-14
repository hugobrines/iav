import pandas as pd
import matplotlib.pyplot as plt
import os

files = os.listdir("data")
# Charger le fichier CSV avec le séparateur approprié et les noms de colonnes
for file in files:
    fichier_csv = "data/" + file
    df = pd.read_csv(fichier_csv, sep=';', header=None, names=['Date', 'Heure', 'Valeur', 'Constante'])

    # Si vous voulez convertir 'Date' en type datetime, faites-le avec dayfirst=True
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

    # Combiner 'Date' et 'Heure' pour obtenir une vraie série temporelle si nécessaire
    df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Heure'])

    # Tracer un graphique de dispersion

    plt.figure(figsize=(10, 6))
    plt.scatter(df['Datetime'], df['Valeur'], label="Valeurs", marker='o')
    plt.xlabel("Temps")
    plt.ylabel("Valeur")
    plt.title("Points individuels avec un écart visible")
    plt.legend()
    plt.grid(True)
    plt.show()
