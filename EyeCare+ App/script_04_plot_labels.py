import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Labels.csv")

# Count for the Label distribution
label_counts = df["Label"].value_counts()
label_counts.plot(kind="bar")

# Plot a Glaucoma Label Distribution Graph
plt.title("Glaucoma Label Distribution")
plt.xlabel("Label")
plt.ylabel("Number of Images")
plt.show()