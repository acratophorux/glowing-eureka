import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
import pickle
import json
import matplotlib.pyplot as plt

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
    
    # Separate features and target
    X = df.filter(regex='^scenario_', axis=1)
    y = df.filter(regex='^solution_', axis=1)
    
    # Scale the input and output columns separately
    input_scaler = MinMaxScaler(feature_range=(-1, 1))
    output_scaler = MinMaxScaler(feature_range=(-1, 1))
    
    X_scaled = input_scaler.fit_transform(X)
    y_scaled = output_scaler.fit_transform(y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=37)
    
    # Train XGBoost model
    xgb_model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=37,
        n_jobs=-1  # Use all available cores
    )
    
    print("Training model...")
    xgb_model.fit(X_train, y_train)
    print("Model training completed!")
    
    # Save model, scaler and feature ranges
    print("Saving model files...")
    with open('xgb_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    
    with open('input_scaler.pkl', 'wb') as f:
        pickle.dump(input_scaler, f)

    with open('output_scaler.pkl', 'wb') as f:
        pickle.dump(output_scaler, f)
    
    with open('feature_ranges.json', 'w') as f:
        json.dump(feature_ranges, f)
    
    print("Model files saved successfully!")
    
    # Calculate and print metrics for both training and test sets
    y_train_pred = xgb_model.predict(X_train)
    y_test_pred = xgb_model.predict(X_test)
    
    print("\nModel Performance:")
    accuracy_df = {}

    for i, col in enumerate(y.columns):
        # Training metrics
        train_mse = np.mean((y_train[:, i] - y_train_pred[:, i]) ** 2)
        train_rmse = np.sqrt(train_mse)
        train_r2 = 1 - (np.sum((y_train[:, i] - y_train_pred[:, i]) ** 2) / np.sum((y_train[:, i] - np.mean(y_train[:, i])) ** 2))
        
        # Test metrics
        test_mse = np.mean((y_test[:, i] - y_test_pred[:, i]) ** 2)
        test_rmse = np.sqrt(test_mse)
        test_r2 = 1 - (np.sum((y_test[:, i] - y_test_pred[:, i]) ** 2) / np.sum((y_test[:, i] - np.mean(y_test[:, i])) ** 2))
        
        accuracy_df[col] = [train_mse, train_r2, test_rmse, test_r2]
        
        print(f"\n{col}:")
        print(f"Training RMSE: {train_rmse:.4f}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Test R²: {test_r2:.4f}")
    
    # Convert the accuracy dictionary to a DataFrame for easier manipulation
    accuracy_df = pd.DataFrame(accuracy_df, index=['Train RMSE', 'Train R²', 'Test RMSE', 'Test R²']).T

    # Save the accuracy_df DataFrame to a CSV file
    accuracy_df.to_csv('model_accuracy_metrics.csv', index=True)

    # plotting the accuracy metrics
    plt.figure(figsize=(12, 6))
    # Set bar width
    bar_width = 0.35
    x = np.arange(len(accuracy_df.index))  # the label locations

    # Plot RMSE
    plt.bar(x - bar_width/2, accuracy_df['Train RMSE'], width=bar_width, label='Train RMSE', alpha=0.7)
    plt.bar(x + bar_width/2, accuracy_df['Test RMSE'], width=bar_width, label='Test RMSE', alpha=0.7)
    plt.title('RMSE Comparison')
    plt.ylabel('RMSE')
    plt.xticks(x, accuracy_df.index, rotation=45)
    plt.legend()

    # Show the plots
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_and_save_model()