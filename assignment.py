import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load dataset
df = pd.read_csv("Employee.csv")

# 1. Data Exploration
print("=== 1. DATA EXPLORATION ===")
print("Shape of dataset:", df.shape)
print("\nUnique values per column:")
for col in df.columns:
  print(f"- {col}: {df[col].nunique()} unique values -> {df[col].unique()}")

print("\nStatistical summary:")
print(df.describe(include="all"))

# 2. Data Cleaning
print("\n=== 2. DATA CLEANING ===")
print("Initial missing values:\n", df.isnull().sum())

# Remove duplicate rows
print("Duplicate rows before:", df.duplicated().sum())
df = df.drop_duplicates()
print("Duplicate rows after:", df.duplicated().sum())

# Replace value 0 in Age with NaN
df["Age"] = df["Age"].replace(0, np.nan)

# Handle missing values (fill numerical with median, categorical with mode)
for col in ["Age", "Salary"]:
  df[col] = df[col].fillna(df[col].median())

for col in ["Company", "Place"]:
  df[col] = df[col].fillna(df[col].mode()[0])

print("Missing values after treatment:\n", df.isnull().sum())

# 3. Data Analysis
print("\n=== 3. DATA ANALYSIS ===")
# Filter data with age > 40 and salary < 5000
filtered_df = df[(df["Age"] > 40) & (df["Salary"] < 5000)]
print(f"Number of employees with Age > 40 and Salary < 5000: {len(filtered_df)}")
print(filtered_df.head())

# Plot the chart with age and salary for filtered data
plt.figure(figsize=(8, 5))
plt.scatter(filtered_df["Age"], filtered_df["Salary"], color="blue", alpha=0.7)
plt.title("Age vs Salary (Age > 40 and Salary < 5000)")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.grid(True)
plt.savefig("age_salary_filtered.png")
plt.close()

# Count the number of people from each place and represent it visually
place_counts = df["Place"].value_counts()
print("\nPeople count by place:\n", place_counts)

plt.figure(figsize=(10, 5))
place_counts.plot(kind="bar", color="teal")
plt.title("Number of Employees from Each Place")
plt.xlabel("Place")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("place_counts.png")
plt.close()

# 4. Data Encoding
print("\n=== 4. DATA ENCODING ===")
encoded_df = pd.get_dummies(
    df, columns=["Company", "Place", "Country"], drop_first=True
)
print("Encoded columns:", encoded_df.columns.tolist())

# 5. Feature Scaling
print("\n=== 5. FEATURE SCALING ===")
scaler_std = StandardScaler()
scaled_std = scaler_std.fit_transform(encoded_df[["Age", "Salary"]])

scaler_mm = MinMaxScaler()
scaled_mm = scaler_mm.fit_transform(encoded_df[["Age", "Salary"]])

print("StandardScaler (first 5 rows):\n", scaled_std[:5])
print("MinMaxScaler (first 5 rows):\n", scaled_mm[:5])