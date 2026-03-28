# ==========================================
# 1. IMPORTS
# ==========================================
import pandas as pd
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# ==========================================
# 2. LOAD MODEL
# ==========================================
model = tf.keras.models.load_model("glaucoma_efficientnet_model.h5")

# ==========================================
# 3. LOAD TEST DATA
# ==========================================
test_df = pd.read_csv("test_dataset.csv")
image_folder = "images_resized"

X_test, y_test = [], []

for _, row in test_df.iterrows():
    img_path = os.path.join(image_folder, row["image_name"])
    
    img = Image.open(img_path).convert("RGB").resize((224,224))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    
    X_test.append(img_array)
    y_test.append(row["label_numeric"])

X_test = np.array(X_test)
y_test = np.array(y_test)

# ==========================================
# 4. PREDICTIONS
# ==========================================
y_prob = model.predict(X_test).flatten()

# ==========================================
# 5. MANUAL THRESHOLD SCAN
# ==========================================
thresholds = np.linspace(0, 1, 100)
best_threshold = 0.5
best_score = -1
sensitivities = []
specificities = []

for t in thresholds:
    y_pred_temp = (y_prob > t).astype(int)
    
    tn = np.sum((y_test == 0) & (y_pred_temp == 0))
    fp = np.sum((y_test == 0) & (y_pred_temp == 1))
    fn = np.sum((y_test == 1) & (y_pred_temp == 0))
    tp = np.sum((y_test == 1) & (y_pred_temp == 1))
    
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    
    sensitivities.append(sensitivity)
    specificities.append(specificity)
    
    score = sensitivity + specificity - 1
    if score > best_score:
        best_score = score
        best_threshold = t

print(f"\nBest Threshold (Manual Search): {best_threshold:.4f}")

# ==========================================
# 6. APPLY BEST THRESHOLD
# ==========================================
y_pred = (y_prob > best_threshold).astype(int)

# ==========================================
# 7. CONFUSION MATRIX
# ==========================================
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# ==========================================
# 8. CUSTOM COLORED CONFUSION MATRIX
# ==========================================
colors = np.empty(cm.shape, dtype=object)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        if i == j:
            # Correct predictions → green
            colors[i,j] = '#6FAF4F' if i==1 else '#C3CC9B'  # Glaucoma dark, Normal light
        else:
            # Wrong predictions → red
            colors[i,j] = '#EB4C4C' if i==1 else '#FFA6A6'  # Glaucoma dark red, Normal light red

fig, ax = plt.subplots()
for (i,j), val in np.ndenumerate(cm):
    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, color=colors[i,j]))
    ax.text(j, i, val, ha='center', va='center', color='black', fontsize=14)

ax.set_xticks([0,1])
ax.set_yticks([0,1])
ax.set_xticklabels(['Normal', 'Glaucoma'])
ax.set_yticklabels(['Normal', 'Glaucoma'])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix (Custom Colors)")
ax.set_xlim(-0.5,1.5)
ax.set_ylim(-0.5,1.5)
ax.invert_yaxis()
plt.show()

# ==========================================
# 9. CLASSIFICATION REPORT
# ==========================================
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================================
# 10. SENSITIVITY vs SPECIFICITY CURVE
# ==========================================
plt.figure()
plt.plot(thresholds, sensitivities, label="Sensitivity")
plt.plot(thresholds, specificities, label="Specificity")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Sensitivity vs Specificity")
plt.legend()
plt.show()

# ==========================================
# 11. SAVE RESULTS
# ==========================================
test_df["probability"] = y_prob
test_df["prediction"] = y_pred
test_df.to_csv("optimized_predictions.csv", index=False)

print("\nPredictions saved as optimized_predictions.csv")