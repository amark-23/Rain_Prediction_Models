# data_modification.py
import joblib
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("train-val.csv")

# Feature Engineering - Combine highly correlated columns
df['MaxTemp'] = (df['MaxTemp'] + df['Temp3pm']) / 2
df['MeanPressure'] = (df['Pressure9am'] + df['Pressure3pm']) / 2

# Convert date and encode month with cyclic encoding
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Month'] = df['Date'].dt.month
df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

# Drop columns with high correlation or redundancy
df = df.drop(columns=[
    'Date', 'Month', 'Pressure9am', 'Pressure3pm',
    'Temp3pm', 'Temp9am', 'RainToday'
], errors='ignore')

print("\033[1;34mInitial cleaned-up structure:\033[0m")
print(df.head())

# KNN Imputation for select columns
print("\n\033[1;34mMissing values before imputation:\033[0m")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

knn_columns = ['Sunshine', 'Evaporation', 'Cloud9am', 'Cloud3pm', 'MeanPressure']
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.difference(knn_columns)
categorical_cols = df.select_dtypes(include=['object']).columns

knn_imputer = KNNImputer(n_neighbors=5)
df[knn_columns] = knn_imputer.fit_transform(df[knn_columns])
df['Cloud9am'] = df['Cloud9am'].round().astype(int)
df['Cloud3pm'] = df['Cloud3pm'].round().astype(int)

# Fill remaining numerical columns with mean
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].mean())

# Fill categorical columns with mode
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\n\033[1;34mMissing values after imputation:\033[0m")
missing_after = df.isnull().sum()
print(missing_after[missing_after > 0] if not missing_after.empty else "\033[1;32mNo missing values remaining.\033[0m")

# One-Hot Encoding for categorical features
cat_cols = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ct = make_column_transformer((ohe, cat_cols), remainder='drop')
ct.set_output(transform='pandas')
df_cat = ct.fit_transform(df)

# Standard Scaling for normal distributions
standard_scaler = StandardScaler()
normal_dist_cols = [
    'MinTemp', 'MaxTemp', 'WindGustSpeed', 'Sunshine', 'WindSpeed9am',
    'Humidity9am', 'Humidity3pm', 'WindSpeed3pm', 'MeanPressure',
    'Cloud9am', 'Cloud3pm', 'Month_sin', 'Month_cos'
]
df[normal_dist_cols] = standard_scaler.fit_transform(df[normal_dist_cols])

# Log-transform + scale Rainfall and Evaporation
df['LogRainfall'] = np.log1p(df['Rainfall'])
df['LogEvaporation'] = np.log1p(df['Evaporation'])
df[['LogRainfall', 'LogEvaporation']] = standard_scaler.fit_transform(
    df[['LogRainfall', 'LogEvaporation']]
)

# Drop original columns that were log-transformed or one-hot encoded
df_num = df.drop(columns=[
    'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm',
    'Evaporation', 'Rainfall'
])

# Final dataset after encoding and scaling
df_cleaned_up = pd.concat([df_num, df_cat], axis=1)

# Train/validation split
X = df_cleaned_up.drop(columns=["RainTomorrow"])
y = df_cleaned_up["RainTomorrow"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

print("\n\033[1;34mFinal training set shape:\033[0m")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape}, y_val:   {y_val.shape}")

# Save training/validation sets to disk
joblib.dump((X_train, X_val, y_train, y_val), "data_splits.joblib")
print("\033[1;32mTraining/Validation data saved to data_splits.joblib\033[0m")