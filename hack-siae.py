# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Hackathon Preparation Kit - SIAE × Datapizza

## Template Notebook Structure

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error

# 2. Load Dataset
df = pd.read_csv('path_to_dataset.csv')
df.head()

# 3. Explore Dataset
print(df.info())
print(df.describe())
print(df.isnull().sum())
sns.heatmap(df.isnull(), cbar=False)

# 4. Data Preprocessing
# - Handle missing values
# - Encode categoricals
# - Feature scaling

# 5. Model Definition & Training
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# 7. Prompt Usage Log
# Prompt: "Suggerisci un modello per classificare testi musicali"
# Risposta AI: "Prova con un RandomForest con TF-IDF"

# 8. Git Commit Placeholder
# git init
# git remote add origin https://github.com/federicodanesin99/hackathon-siae
# git add .
# git commit -m "Initial commit"
# git push origin main
