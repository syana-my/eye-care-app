import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("glaucoma_clean_data.csv")
if "Unnamed: 5" in df.columns:
	df = df.drop(columns=["Unnamed: 5"])

# Get unique patient IDs
patients = df["patient"].unique()

# Split patients into train/test
train_patients, test_patients = train_test_split(
	patients,
	test_size=0.2,
	random_state=42,
)

# Select images belonging to those patients
train_df = df[df["patient"].isin(train_patients)]
test_df = df[df["patient"].isin(test_patients)]

print("Training images:", len(train_df))
print("Testing images:", len(test_df))
print("Training patients:", train_df["patient"].nunique())
print("Testing patients:", test_df["patient"].nunique())

train_df.to_csv("train_dataset.csv", index=False)
test_df.to_csv("test_dataset.csv", index=False)
print("Datasets saved successfully.")