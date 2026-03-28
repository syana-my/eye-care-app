import pandas as pd

df = pd.read_csv("Labels.csv")

# Drop unwanted column
if "Unnamed: 4" in df.columns:
    df = df.drop(columns=["Unnamed: 4"])

# Convert label
df["label"] = df["Label"].map({"GON+": 1, "GON-": 0})

# ✅ FILTER QUALITY (THIS WAS MISSING)
df = df[df["quality_score"] >= 5]

# Save clean dataset
df.to_csv("glaucoma_clean_data.csv", index=False)

print("Clean dataset saved.")