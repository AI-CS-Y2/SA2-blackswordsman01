import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, roc_curve
from sklearn import tree
import graphviz

# Load the dataset
file_path = 'anime.csv'  
df = pd.read_csv(file_path)

# Data Preprocessing
df = df[['Type', 'Episodes']]  

# Drop rows with missing values
df = df.dropna(subset=['Type', 'Episodes'])

# Convert 'Episodes' to numeric values
df['Episodes'] = pd.to_numeric(df['Episodes'], errors='coerce')
df = df.dropna(subset=['Episodes'])

# Convert 'Type' column to binary: 'Movie' -> 1, others -> 0
df['movie'] = (df['Type'] == 'Movie').astype(int)

# Features and target
X = df[['Episodes']]  
y = df['movie']  

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree model
clf = DecisionTreeClassifier(random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

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
plt.title("Decision Tree ROC Curve", fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.grid()
plt.show()

# Visualize the Decision Tree
dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=['Episodes'],
                                class_names=['Non-Movie', 'Movie'],
                                filled=True, rounded=True,
                                special_characters=True)

graph = graphviz.Source(dot_data)
graph.render("decision_tree_visual")
graph.view()

