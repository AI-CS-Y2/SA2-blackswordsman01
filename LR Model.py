# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, roc_curve

# Load the dataset
file_path = 'anime.csv'  
df = pd.read_csv(file_path)

# Keep only the relevant columns
df = df[['Type', 'Episodes']]

# Drop rows with missing values
df = df.dropna(subset=['Type', 'Episodes'])

# Convert 'Episodes' to numeric values
df['Episodes'] = pd.to_numeric(df['Episodes'], errors='coerce')
df = df.dropna(subset=['Episodes'])

# Convert the 'Type' column to binary: 'Movie' -> 1, others -> 0
df['movie'] = (df['Type'] == 'Movie').astype(int)

# Features and target
X = df[['Episodes']]
y = df['movie']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Logistic Regression model 
log_reg = LogisticRegression(C=0.5, random_state=42)

# Train the model
log_reg.fit(X_train_scaled, y_train)

# Predictions
y_pred = log_reg.predict(X_test_scaled)
y_prob = log_reg.predict_proba(X_test_scaled)[:, 1]  

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
loss = log_loss(y_test, y_prob)

# Output performance metrics
print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'ROC AUC: {roc_auc:.4f}')
print(f'Log Loss: {loss:.4f}')

# Plot the ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
plt.title("Logistic Regression ROC Curve", fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.grid()
plt.show()