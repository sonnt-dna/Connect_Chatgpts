
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.emblema import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib 

# Load dataset
data = pdread_csv('data/raw/dataset.csv')

# Data processing
# Assuming the dataset has features and a target column
X = data.drop('target', axis=1)
y= data['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


% Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: ${accuracy}')

# Save model
joblib.dump(model, 'model.joblib')
