import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
import pickle
import json

def train_and_save_model(data_path='vsc_hvdc_dataset.csv'):
    # Load the data
    df = pd.read_csv(data_path)
    
    # Get feature ranges for GUI validation
    feature_ranges = {}
    for col in df.columns:
        if col.startswith('scenario_'):
            feature_ranges[col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max())
            }
    
    # Scale the numerical columns
    cols_to_scale = ['scenario_P1', 'scenario_Qg1', 'scenario_Qg2', 'scenario_AC1_Nom_1', 'scenario_AC1_Nom_2', 'scenario_AC1_Nom_3', 'scenario_AC1_Nom_4', 'scenario_AC1_Nom_5', 'scenario_AC2_Nom_1', 'scenario_AC2_Nom_2', 'scenario_AC2_Nom_3', 'scenario_AC2_Nom_4', 'scenario_AC2_Nom_5', 'scenario_DC_Nom_2', 'solution_N_vw1', 'solution_N_vw2', 'solution_Qcab1_interm', 'solution_Qcab2_interm', 'solution_Qf1_interm', 'solution_Qf2', 'solution_Rdc', 'solution_Vac1', 'solution_Vac2', 'solution_Vdc1', 'solution_w1', 'solution_w2']
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    
    # Separate features and target
    X = df.filter(regex='^scenario_', axis=1)
    y = df.filter(regex='^solution_', axis=1)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=37)
    
    # Train XGBoost model
    xgb_model = MultiOutputRegressor(
        XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=37,
            n_jobs=-1  # Use all available cores
        )
    )
    
    print("Training model...")
    xgb_model.fit(X_train, y_train)
    print("Model training completed!")
    
    # Save model, scaler and feature ranges
    print("Saving model files...")
    with open('xgb_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open('feature_ranges.json', 'w') as f:
        json.dump(feature_ranges, f)
    
    print("Model files saved successfully!")
    
    # Calculate and print metrics
    y_pred = xgb_model.predict(X_test)
    
    print("\nModel Performance:")
    for i, col in enumerate(y.columns):
        mse = np.mean((y_test.iloc[:, i] - y_pred[:, i]) ** 2)
        rmse = np.sqrt(mse)
        r2 = 1 - (np.sum((y_test.iloc[:, i] - y_pred[:, i]) ** 2) / 
                  np.sum((y_test.iloc[:, i] - y_test.iloc[:, i].mean()) ** 2))
        
        print(f"\n{col}:")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")

if __name__ == "__main__":
    train_and_save_model()