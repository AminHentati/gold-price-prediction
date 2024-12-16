import pandas as pd
import os

def preprocess_data(input_path, output_path):
    # Charger les données
    df = pd.read_csv(input_path)
    df = df.iloc[2:].reset_index(drop=True)

    # Convertir les colonnes numériques
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Nettoyage des données
    df = df.dropna()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print("Données prétraitées sauvegardées.")

if __name__ == "__main__":
    preprocess_data("data/raw/gold_data.csv", "data/processed/cleaned_gold_data.csv")
