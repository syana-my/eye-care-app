import pandas as pd

# Load dataset
df = pd.read_csv("Labels.csv")

# Display statistics of the quality score column 
print("\nQuality score statistics:\n")
print(df["quality_score"].describe())
print("\n")