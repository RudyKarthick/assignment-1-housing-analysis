# cleaning_rudy.py

import pandas as pd

# Load the original dataset
housing = pd.read_csv("housing.csv")
print("✅ Dataset loaded successfully.")

# === Step 1: Handle missing values in 'total_bedrooms' ===
# A few rows are missing data in this column. We’ll fill those gaps with the median value.
missing_count = housing['total_bedrooms'].isnull().sum()
print(f"🔍 Missing entries in 'total_bedrooms': {missing_count}")

# Calculate the median and fill the missing values
median_bedrooms = housing['total_bedrooms'].median()
housing['total_bedrooms'].fillna(median_bedrooms, inplace=True)
print("✅ Filled missing 'total_bedrooms' values with the median.")

# Double-check that there are no more missing values
assert housing['total_bedrooms'].isnull().sum() == 0

# === Step 2: One-hot encode the 'ocean_proximity' column ===
# This column contains categories like 'INLAND' and 'NEAR OCEAN'.
# We'll convert it into multiple binary columns so models can work with it.

print("🔄 Applying one-hot encoding to 'ocean_proximity'...")
housing_encoded = pd.get_dummies(housing, columns=['ocean_proximity'])

# Show the new column names after encoding
print("✅ One-hot encoding complete. New columns:")
print(housing_encoded.columns.tolist())

# Optional: Save the cleaned dataset
housing_encoded.to_csv("housing_cleaned_by_rudy.csv", index=False)
print("💾 Cleaned dataset saved to housing_cleaned_by_rudy.csv")
