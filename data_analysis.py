# data_analysis.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("train-val.csv")  # Make sure this file is in your working directory

# Drop 'ID' column if it exists
df_no_id = df.drop(columns=['ID'], errors='ignore')

# 2a. Number of samples and features
num_samples, num_features = df.shape
print("\033[1;34m----- 2a: Number of Samples and Features -------------------------------------------------------\033[0m")
print(f"\033[1;32mNumber of samples:\033[0m {num_samples}")
print(f"\033[1;32mNumber of features:\033[0m {num_features}")
print("\n")

# 2b, 2c. Feature labels and data types
print("\033[1;34m----- 2b,c: Feature Labels & Data Types -----------------------------------------------------\033[0m")
df_info = pd.DataFrame({
    "Label": df.columns,
    "Data Type": df.dtypes.values
})
print(df_info)
print("\n")

# 2d. Unique categories in categorical features
print("\033[1;34m----- 2d: Unique Categories in Categorical Features --------------------------------------------\033[0m")
categorical_columns = df_no_id.select_dtypes(include=['object']).columns
for col in categorical_columns:
    print(f"\033[1;36mFeature:\033[0m {col}")
    print("\033[1;35mUnique values:\033[0m \n", df_no_id[col].unique())
    print("\033[1;35mNumber of unique values:\033[0m", df_no_id[col].nunique())
    print("\033[1;37m" + "-" * 80 + "\033[0m")
print("\n")

# 2e. Value counts per category
print("\033[1;34m----- 2e: Sample Distribution per Category ------------------------------------------------------------\033[0m")
for col in categorical_columns:
    print(f"\033[1;36mDistribution for feature '{col}':\033[0m")
    print(df_no_id[col].value_counts())
    print("\033[1;37m" + "-" * 80 + "\033[0m")
print("\n")

# 2f. Correlation between numerical features
print("\033[1;34m----- 2f: Correlation Between Numerical Features ------------------------------------------------------\033[0m")

numerical_columns = df_no_id.select_dtypes(include=['float64', 'int64','int32']).columns
correlation_matrix = df_no_id[numerical_columns].corr()
high_corr_pairs = [
        (correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iat[i, j])
        for i in range(len(correlation_matrix.columns)) for j in range(i)
        if correlation_matrix.iat[i, j] > 0.8
    ]
if high_corr_pairs:
  print("\033[1;35m----- Highly Correlated Feature Pairs (Correlation > 0.8): -------\033[0m \n")
  for pair in high_corr_pairs:
    print(f"  {pair[0]} and {pair[1]}: {pair[2]:.2f}")
else:
    print("No highly correlated pairs found.\n")

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Correlation Heatmap of Numerical Features", fontsize=16)
plt.show()

# 2g. Additional insights
print("\033[1;34m \n----- 2g: Additional Useful Information ---------------------------------------------------------------------\033[0m\n")

# Missing values
print("\033[1;34m----- Missing Values ----------------------------------------------------------------------\033[0m")
missing_values = df_no_id.isnull().sum()
if missing_values[missing_values > 0].any():
    print(missing_values[missing_values > 0])
else:
    print("\033[1;32mNo missing values found!\033[0m")

# Summary statistics
print(f"\n\033[1;34m----- Summary Statistics for Numerical Features ------------------------------------------------\033[0m")
print(df_no_id[numerical_columns].describe())
print("\n")

# Histograms
print("\033[1;34m----- Distribution of Numerical Features ---------------------------------------------------------------\033[0m")
plt.figure(figsize=(20, 10))
df_no_id[numerical_columns].hist(bins=30, figsize=(15, 10), color='lightblue', edgecolor='black')
plt.suptitle("Distribution of Numerical Features", fontsize=20)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
