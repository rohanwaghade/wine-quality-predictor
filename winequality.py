#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder


# In[ ]:


# Load dataset
df = pd.read_csv("winequality.csv")

# Display basic info
print("Dataset shape:", df.shape)
print("Column names:", df.columns.tolist())
print(df.head())


# In[ ]:


print("\nMissing values:\n", df.isnull().sum())


# In[ ]:


# Target variable: Convert quality to binary (good: 1 if >=7, else 0)
df["quality_binary"] = df["quality"].apply(lambda x: 1 if x >= 7 else 0)


# In[ ]:


# Features and target
X = df.drop(["quality", "quality_binary"], axis=1)
y = df["quality_binary"]


# In[ ]:


# Encode non-numeric 'type' column (if present)
if 'type' in df.columns:
    df['type'] = LabelEncoder().fit_transform(df['type'])

# Features and target
X = df.drop(["quality", "quality_binary"], axis=1)
y = df["quality_binary"]


# In[ ]:


# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[ ]:


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# In[ ]:


# Handle missing values
X = X.fillna(X.mean())  # or use dropna() if preferred


# In[ ]:


X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# In[ ]:


# Evaluate
y_pred = model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# In[ ]:


# Feature importance
feature_importance = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance}).sort_values(by='Importance', ascending=False)


# In[ ]:


print("\nTop Features:\n", importance_df)


# In[ ]:


# Plot feature importances
plt.figure(figsize=(10,6))
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="magma")
plt.title("Feature Importance in Wine Quality Prediction")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()


# In[ ]:


# Save model and scaler
joblib.dump(model, "wine_quality_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel and scaler saved successfully.")


# In[ ]:




