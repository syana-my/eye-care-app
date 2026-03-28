import pandas as pd

# Load dataset
df = pd.read_csv("Labels.csv")
print("\nDataset successfully loaded.\n")

# Display the first 5 rows of dataset
print("First 5 rows of dataset:\n")
print(df.head())

# Display dataset shape in (row,column) format
print("\nDataset shape:")
print(df.shape)

# To check for the column names of dataset
print("\nColumn names:")
print(df.columns)

# Display distribution of Label column in the dataset
print("\nLabel distribution:\n")
print(df["Label"].value_counts())
print("\n")