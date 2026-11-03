# ML-Practical
# Practical No. 1
# Title: Feature Engineering – Handling Missing Values, Encoding and Scaling

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load Dataset
df = pd.read_csv("train.csv")

print("Original Dataset")
print(df.head())

# Check Missing Values
print("\nMissing Values")
print(df.isnull().sum())

# Handle Missing Values
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Encode Categorical Variables
encoder = LabelEncoder()

df["Sex"] = encoder.fit_transform(df["Sex"])
df["Embarked"] = encoder.fit_transform(df["Embarked"])

# Scale Numerical Attributes
scaler = StandardScaler()

df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])

print("\nPreprocessed Dataset")
print(df[["Age", "Fare", "Sex", "Embarked"]].head())

