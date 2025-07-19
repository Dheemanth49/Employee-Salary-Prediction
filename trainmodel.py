

import pandas as pd
import numpy as np
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Model Imports
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor


def load_and_preprocess_data(csv_path:str):
    """Loads and preprocesses data, returning the dataframe and encoders."""
    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)

    # Coerce columns to numeric, dropping rows with errors
    for col in ['Age', 'Years of Experience', 'Salary']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    label_encoders = {}
    categorical_columns = ['Gender', 'Education Level', 'Job Title']
    
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        
    return df, label_encoders

def main():
    """Main function to train, compare, and save the best model."""
    feature_columns = ['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']
    target_column = 'Salary'

    # 1 Load Data
    path=r"Salary_Data.csv"
    df, label_encoders = load_and_preprocess_data(path)

    # 2 Split Data
    X = df[feature_columns]
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3 Scale Numerical Features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 4 Define Models to Compare
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'KNeighbors': KNeighborsRegressor(n_neighbors=5),
        'SVR': SVR(kernel='rbf'),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42, learning_rate=0.1, max_depth=5)
    }

    # 5 Train and Evaluate Models
    results = []
    best_model = None
    best_model_name = ""
    best_r2 = -1

    for name, model in models.items():
        print(f"--- Training {name} ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        results.append({'Model': name, 'R² Score': r2, 'RMSE': rmse})
        print(f"R² Score: {r2:.4f}, RMSE: {rmse:.2f}\n")

        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_model_name = name

    # 6 Display Comparison and Select Best Model
    results_df = pd.DataFrame(results).sort_values(by='R² Score', ascending=False)
    print("--- Model Comparison ---")
    print(results_df)
    
    print(f"\n Best performing model is '{best_model_name}' with an R² score of {best_r2:.4f}")

    # 7 Save the Best Model, Encoders, and Scaler
    model_data = {
        'model': best_model,
        'label_encoders': label_encoders,
        'scaler': scaler,
        'feature_columns': feature_columns
    }
    joblib.dump(model_data, r'salary_model.pkl')
    print(f"\nBest model ('{best_model_name}') and preprocessors saved to salary_model.pkl")

    # 8 Save Category Mappings / encoding
    category_info = {}
    for col, encoder in label_encoders.items():
        category_info[col] = {
            'classes': encoder.classes_.tolist(),
        }
    
    with open(r'category_mappings.json', 'w') as f:
        json.dump(category_info, f, indent=2)
    print("Category mappings saved to category_mappings.json")

if __name__ == "__main__":
    main()