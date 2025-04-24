#!/usr/bin/env python
# coding: utf-8

# In[1]:


# wine_quality_prediction.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def load_and_preprocess_data(filepath):
    """Load and preprocess the wine quality dataset"""
    df = pd.read_csv("winequality.csv")
    
    # Convert wine type to numerical (red=0, white=1)
    df['type'] = LabelEncoder().fit_transform(df['type'])

    # Handle missing values
    df = df.fillna(df.mean(numeric_only=True))

    # Create binary target (good wine if quality >= 7)
    df['quality_binary'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)
    
    return df

def train_model(X_train, y_train):
    """Train the Random Forest classifier"""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    print("Model Evaluation:")
    print("=" * 50)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    return y_pred

def plot_feature_importance(model, feature_names):
    """Plot feature importances"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.title("Feature Importances in Wine Quality Prediction")
    plt.bar(range(len(feature_names)), importances[indices], align="center")
    plt.xticks(range(len(feature_names)), feature_names[indices], rotation=90)
    plt.tight_layout()
    plt.show()

def save_model(model, scaler, model_path="wine_quality_model.pkl", scaler_path="wine_quality_scaler.pkl"):
    """Save trained model and scaler"""
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

def predict_new_sample(model, scaler, sample_data):
    """Predict wine quality for a new sample"""
    sample_scaled = scaler.transform(sample_data)
    prediction = model.predict(sample_scaled)
    probability = model.predict_proba(sample_scaled)
    return prediction, probability

def main():
    # Load and preprocess data
    df = load_and_preprocess_data("winequality.csv")

    # Prepare features and target
    drop_cols = ['quality', 'quality_binary']
    if 'id' in df.columns:
        drop_cols.append('id')
    X = df.drop(drop_cols, axis=1)
    y = df['quality_binary']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = train_model(X_train_scaled, y_train)

    # Evaluate model
    evaluate_model(model, X_test_scaled, y_test)

    # Plot feature importance
    plot_feature_importance(model, X.columns.values)

    # Save model and scaler
    save_model(model, scaler)

    # Predict on an example sample
    example_wine = pd.DataFrame([[7.4, 0.7, 0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4, 0]], 
                                 columns=X.columns)
    quality, prob = predict_new_sample(model, scaler, example_wine)
    print("\nExample Prediction:")
    print("=" * 50)
    print(f"Predicted Quality: {'Good' if quality[0] else 'Not Good'}")
    print(f"Probability: {prob[0][1]*100:.2f}%")

if __name__ == "__main__":
    main()


# In[ ]:




