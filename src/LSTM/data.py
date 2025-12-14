import pandas as pd

# Fonctions pour charger et traiter les données
def load_data_from_csv(filename):
    """Charge les données à partir d'un fichier CSV"""
    df = pd.read_csv(filename, sep=';', header=None)
    df.columns = ['Date', 'Heure', 'Valeur', 'Constante']
    df['Datetime'] = pd.to_datetime(df['Date'] + " " + df['Heure'], format='%d/%m/%Y %H:%M:%S')
    df.sort_values('Datetime', inplace=True)
    
    # Extraction des valeurs
    data = df['Valeur'].values
    times = df['Datetime'].values

    return data, times, df

def generate_stream_from_csv(filename, start_idx=0):
    """Lit le CSV et retourne un générateur de (datetime, valeur) à partir d'un certain indice."""
    df = pd.read_csv(filename, sep=';', header=None)
    df.columns = ['Date', 'Heure', 'Valeur', 'Constante']
    df['Datetime'] = pd.to_datetime(df['Date'] + " " + df['Heure'], format='%d/%m/%Y %H:%M:%S')
    df.sort_values('Datetime', inplace=True)
    
    # Commencer à partir de l'indice spécifié
    df_from_start = df.iloc[start_idx:].reset_index(drop=True)
    
    for index, row in df_from_start.iterrows():
        yield row['Datetime'], row['Valeur']
