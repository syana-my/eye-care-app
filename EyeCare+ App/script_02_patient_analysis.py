import pandas as pd

# Load dataset
df = pd.read_csv("Labels.csv")

# Display total number of dataset
print("\nTotal number of images:")
print(len(df))

# Display total number of patient
print("\nTotal unique patients:")
print(df["Patient"].nunique())

# To check duplication of images for the first 5 of patient
print("\nImages per patient:")
print(df.groupby("Patient").size().head())