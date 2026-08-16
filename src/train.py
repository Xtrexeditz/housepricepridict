import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def train_model():
    print("Loading data...")
    data_path = 'data/housing_data.csv'
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run generate_data.py first.")
        
    df = pd.read_csv(data_path)
    
    # Separate features and target
    X = df.drop(columns=['Price'])
    y = df['Price']
    
    # Define features
    num_features = ['SquareFeet', 'Bedrooms', 'Bathrooms', 'YearBuilt', 'Floors', 'Garage']
    cat_features = ['Neighborhood', 'PropertyType']
    
    # Preprocessing pipelines
    num_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ])
    
    # Create complete training pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    
    # Fit the pipeline
    print("Training Random Forest Regressor...")
    model_pipeline.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model_pipeline.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Model Evaluation ---")
    print(f"Mean Absolute Error (MAE): Rs. {mae:,.2f}")
    print(f"Mean Squared Error (MSE): {mse:,.2f}")
    print(f"Root Mean Squared Error (RMSE): Rs. {rmse:,.2f}")
    print(f"R-squared (R2) Score: {r2:.4f}")
    print("------------------------")
    
    # Save the pipeline
    os.makedirs('models', exist_ok=True)
    pipeline_path = 'models/house_price_pipeline.joblib'
    joblib.dump(model_pipeline, pipeline_path)
    print(f"\nModel pipeline successfully saved to {pipeline_path}")
    
    # Print feature importances
    onehot_cols = list(model_pipeline.named_steps['preprocessor']
                       .named_transformers_['cat']
                       .named_steps['onehot']
                       .get_feature_names_out(cat_features))
    feature_names = num_features + onehot_cols
    importances = model_pipeline.named_steps['regressor'].feature_importances_
    
    print("\nFeature Importances:")
    for name, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {importance:.4f}")

if __name__ == '__main__':
    train_model()
