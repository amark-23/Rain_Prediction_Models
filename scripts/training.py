# training.py

import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

# Load training and validation sets
X_train, X_val, y_train, y_val = joblib.load("data_splits.joblib")

# Initialize classifiers with default hyperparameters
classifiers = {
    "Naive Bayes": GaussianNB(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "MLP (Neural Net)": MLPClassifier(hidden_layer_sizes=(10,)),
    "Support Vector Machine": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Train all classifiers
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    print(f"{name} classifier trained.")

# Predict on validation set
f1_scores = {}
accuracies = {}

for name, clf in classifiers.items():
    y_pred = clf.predict(X_val)

    print(f"\n{name} predictions on validation set:")
    print(y_pred)

    f1 = f1_score(y_val, y_pred, average="weighted")
    accuracy = accuracy_score(y_val, y_pred)

    f1_scores[name] = f1
    accuracies[name] = accuracy

    print(f"{name} - F1 Score on validation set: {f1:.2f}")
    print(f"{name} - Accuracy on validation set: {accuracy:.2f}")

# Plot F1 scores and accuracies
plt.figure(figsize=(12, 6))

# F1 Score bar plot
plt.subplot(1, 2, 1)
plt.bar(f1_scores.keys(), f1_scores.values(), color='skyblue')
plt.xticks(rotation=45, ha="right")
plt.xlabel("Classifier")
plt.ylabel("F1 Score")
plt.title("Classifier Comparison (F1 Score)")

# Accuracy bar plot
plt.subplot(1, 2, 2)
plt.bar(accuracies.keys(), accuracies.values(), color='salmon')
plt.xticks(rotation=45, ha="right")
plt.xlabel("Classifier")
plt.ylabel("Accuracy")
plt.title("Classifier Comparison (Accuracy)")

plt.tight_layout()
plt.show()

# Save trained classifiers
joblib.dump(classifiers, "classifiers.joblib")
print("\033[1;32mAll trained classifiers saved to classifiers.joblib\033[0m")