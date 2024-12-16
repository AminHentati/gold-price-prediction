import pandas as pd
import joblib
import mlflow
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

def train_model(data_path, model_path):
    mlflow.set_experiment("gold-price-prediction")
    
    # Charger les données
    df = pd.read_csv(data_path)
    X = df[['Open', 'High', 'Low', 'Volume']]
    y = df['Close']

    # Train-test split
    train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=0)

    with mlflow.start_run():
        # Initialiser et entraîner le modèle
        model = LinearRegression()
        model.fit(train_X, train_y)

        # Évaluation
        val_predictions = model.predict(val_X)
        mae = mean_absolute_error(val_y, val_predictions)
        rmse = mean_squared_error(val_y, val_predictions, squared=False)

        # Log avec MLflow
        mlflow.log_param("model", "LinearRegression")
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)

        # Créer le dossier s'il n'existe pas
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Sauvegarde du modèle
        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model, "model")
        print(f"Modèle sauvegardé avec MAE: {mae}, RMSE: {rmse}")

if __name__ == "__main__":
    train_model("data/processed/cleaned_gold_data.csv", "models/gold_price_model.pkl")
