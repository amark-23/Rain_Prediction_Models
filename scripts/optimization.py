import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import joblib

# Load classifiers trained previously
classifiers = joblib.load("classifiers.joblib")

# Load data splits as well
X_train, X_val, y_train, y_val = joblib.load("data_splits.joblib")

# Load test set
df_test = pd.read_csv("datasets/test.csv")

# Feature engineering
df_test['MaxTemp'] = (df_test['MaxTemp'] + df_test['Temp3pm']) / 2
df_test['MeanPressure'] = (df_test['Pressure9am'] + df_test['Pressure3pm']) / 2
df_test['Date'] = pd.to_datetime(df_test['Date'], errors='coerce')
df_test['Month'] = df_test['Date'].dt.month
df_test['Month_sin'] = np.sin(2 * np.pi * df_test['Month'] / 12)
df_test['Month_cos'] = np.cos(2 * np.pi * df_test['Month'] / 12)

# Drop redundant columns
df_test.drop(columns=[
    'Date', 'Month', 'Pressure9am', 'Pressure3pm', 'Temp3pm',
    'Temp9am', 'RainToday'
], inplace=True, errors='ignore')

# Imputation
knn_columns = ['Sunshine', 'Evaporation', 'Cloud9am', 'Cloud3pm', 'MeanPressure']
numeric_cols = df_test.select_dtypes(include=['float64', 'int64']).columns.difference(knn_columns)
categorical_cols = df_test.select_dtypes(include=['object']).columns

knn_imputer = KNNImputer(n_neighbors=5)
df_test[knn_columns] = knn_imputer.fit_transform(df_test[knn_columns])
df_test['Cloud9am'] = df_test['Cloud9am'].round().astype(int)
df_test['Cloud3pm'] = df_test['Cloud3pm'].round().astype(int)

for col in numeric_cols:
    df_test[col] = df_test[col].fillna(df_test[col].mean())
for col in categorical_cols:
    df_test[col] = df_test[col].fillna(df_test[col].mode()[0])

# Encoding
cat_cols = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ct = make_column_transformer((ohe, cat_cols), remainder='drop')
ct.set_output(transform='pandas')
df_test_cat = ct.fit_transform(df_test)

# Scaling
standard_scaler = StandardScaler()
normal_dist_cols = [
    'MinTemp', 'MaxTemp', 'WindGustSpeed', 'Sunshine',
    'WindSpeed9am', 'Humidity9am', 'Humidity3pm',
    'WindSpeed3pm', 'MeanPressure', 'Cloud9am', 'Cloud3pm'
]
df_test[normal_dist_cols] = standard_scaler.fit_transform(df_test[normal_dist_cols])

df_test['LogRainfall'] = np.log1p(df_test['Rainfall'])
df_test['LogEvaporation'] = np.log1p(df_test['Evaporation'])
df_test[['LogRainfall', 'LogEvaporation']] = standard_scaler.fit_transform(df_test[['LogRainfall', 'LogEvaporation']])

df_test_num = df_test.drop(columns=['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'Evaporation', 'Rainfall'])
df_test_final = pd.concat([df_test_num, df_test_cat], axis=1)


# Select best model based on previous F1 scores
f1_scores = {}
for name, clf in classifiers.items():
    y_pred = clf.predict(X_val)
    f1_scores[name] = f1_score(y_val, y_pred, average='weighted')

best_model_name = max(f1_scores, key=f1_scores.get)
best_model = classifiers[best_model_name]

print(f"The best performing model is: {best_model_name}")

# Predict on test set
test_predictions = best_model.predict(df_test_final)
print(test_predictions)

# Save predictions to CSV
results = pd.DataFrame({
    "ID": df_test["ID"],
    "Prediction": test_predictions
})
results.to_csv("predictions.csv", index=False)
print("Predictions saved to predictions.csv")

# Grid Search Optimization
best_classifiers = {}

param_grids = {
    'Naive Bayes': {
        'var_smoothing': [1e-9, 1e-8, 1e-7]
    },
    'KNeighborsClassifier': {
        'n_neighbors': [3, 5],
        'weights': ['uniform', 'distance']
    },
    'LogisticRegression': {
        'C': [0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs']
    },
    'MLP (Neural Net)': {
        'hidden_layer_sizes': [(50,), (100,)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001],
        'learning_rate': ['constant']
    },
    'Support Vector Machine': {
        'C': [1, 10],
        'kernel': ['rbf'],
        'gamma': ['scale']
    },
    'Decision Tree': {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 20],
        'min_samples_split': [2, 5]
    },
    'Random Forest': {
        'n_estimators': [50, 100],
        'criterion': ['gini'],
        'max_depth': [None, 20],
        'min_samples_split': [2, 5]
    }
}

for name, clf in classifiers.items():
    print(f"Optimizing classifier: {name}")
    grid_search = GridSearchCV(
        estimator=clf,
        param_grid=param_grids[name],
        scoring='f1_weighted',
        cv=5,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_classifiers[name] = grid_search.best_estimator_

    print(f"Best parameters for {name}: {grid_search.best_params_}")
    print(f"Best F1 Score for {name}: {grid_search.best_score_}\n")

# Validate optimized models
validation_predictions = {}
for name, clf in best_classifiers.items():
    validation_predictions[name] = clf.predict(X_val)
    print(f"Predictions on validation set for classifier {name} completed.")

# Evaluate and compare
f1_scores = {}
for name, predictions in validation_predictions.items():
    f1 = f1_score(y_val, predictions, average='weighted')
    f1_scores[name] = f1
    print(f"F1 Score for {name}: {f1:.4f}")

# Plot F1 scores
plt.figure(figsize=(10, 6))
plt.bar(f1_scores.keys(), f1_scores.values(), color='skyblue')
plt.xlabel('Classifiers')
plt.ylabel('F1 Score')
plt.title('Classifier Performance Comparison (F1 Score)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Best model result
best_model_name = max(f1_scores, key=f1_scores.get)
print(f"The best model after optimization is {best_model_name} with F1 Score: {f1_scores[best_model_name]:.4f}")
