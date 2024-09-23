#!/usr/bin/env python
# coding: utf-8

# In[15]:


pip install ucimlrepo


# In[16]:


from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 
  
# data (as pandas dataframes) 
X = heart_disease.data.features 
y = heart_disease.data.targets 
  
# metadata 
print(heart_disease.metadata) 
  
# variable information 
print(heart_disease.variables) 


# In[17]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Preprocess features (X) and labels (y)
# Handle missing values (if any)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Normalize/Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Print shape to confirm
print(f"Train data shape: {X_train.shape}, Test data shape: {X_test.shape}")


# In[18]:


# Let's print the first few rows of the data for reference
print(X.head())

# In this case, the dataset is already well-structured, but you can add domain-specific features if needed.
# For now, we'll proceed without any additional feature engineering.


# In[19]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Initialize the models
log_reg = LogisticRegression(random_state=42)
rf = RandomForestClassifier(random_state=42)

# Train Logistic Regression Model
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)

# Train Random Forest Model
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Evaluate the performance of both models
log_reg_acc = accuracy_score(y_test, y_pred_log_reg)
rf_acc = accuracy_score(y_test, y_pred_rf)

print("Logistic Regression Accuracy: ", log_reg_acc)
print("Random Forest Accuracy: ", rf_acc)

# Confusion Matrix and Classification Report for Random Forest
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))


# In[20]:


def treatment_recommendation(patient_data):
    # Assume Random Forest is our final model for prediction
    prediction = rf.predict(patient_data)
    
    if prediction == 1:
        return "High risk of heart disease. Recommend further tests and immediate treatment."
    else:
        return "Low risk of heart disease. Recommend regular check-ups and preventive care."

# Example usage with a test patient
sample_patient = X_test[0].reshape(1, -1)
print("Treatment Recommendation:", treatment_recommendation(sample_patient))


# In[21]:


import matplotlib.pyplot as plt
import numpy as np

# Feature importance from Random Forest
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importance (Random Forest)")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.tight_layout()
plt.show()

# Print important features
for i in indices:
    print(f"{X.columns[i]}: {importances[i]}")


# In[25]:


import numpy as np

# Helper function to validate float inputs
def get_float_input(prompt):
    while True:
        try:
            value = float(input(prompt))
            return value
        except ValueError:
            print("Invalid input. Please enter a valid number.")

# Helper function to validate integer inputs
def get_int_input(prompt, valid_options=None):
    while True:
        try:
            value = int(input(prompt))
            if valid_options and value not in valid_options:
                print(f"Invalid input. Please choose from {valid_options}.")
            else:
                return value
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

def cli_for_doctors():
    print("Welcome to the DocAssist Medical Decision Support System!")
    
    # Get inputs for all relevant features with validation
    age = get_float_input("Enter patient's age: ")
    sex = get_int_input("Enter patient's sex (1 = male, 0 = female): ", valid_options=[0, 1])
    cp = get_int_input("Enter chest pain type (0-3): ", valid_options=[0, 1, 2, 3])
    trestbps = get_float_input("Enter resting blood pressure (in mm Hg): ")
    chol = get_float_input("Enter serum cholesterol in mg/dl: ")
    fbs = get_int_input("Fasting blood sugar > 120 mg/dl? (1 = true, 0 = false): ", valid_options=[0, 1])
    restecg = get_int_input("Resting electrocardiographic results (0-2): ", valid_options=[0, 1, 2])
    thalach = get_float_input("Enter maximum heart rate achieved: ")
    exang = get_int_input("Exercise induced angina (1 = yes, 0 = no): ", valid_options=[0, 1])
    oldpeak = get_float_input("Enter ST depression induced by exercise relative to rest: ")
    slope = get_int_input("Slope of the peak exercise ST segment (0-2): ", valid_options=[0, 1, 2])
    ca = get_int_input("Number of major vessels (0-4) colored by fluoroscopy: ", valid_options=[0, 1, 2, 3, 4])
    thal = get_int_input("Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect): ", valid_options=[0, 1, 2])

    # Create the patient data array
    patient_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
    
    # Scale the input data using the same scaler used in preprocessing
    patient_data_scaled = scaler.transform(patient_data)
    
    # Get treatment recommendation
    recommendation = treatment_recommendation(patient_data_scaled)
    
    print("Recommendation:", recommendation)

# Run the CLI
cli_for_doctors()


# In[ ]:




