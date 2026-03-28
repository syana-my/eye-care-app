# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================
import pandas as pd
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt

print("TensorFlow Version:", tf.__version__)

# ==========================================
# 2. LOAD DATASET
# ==========================================
df = pd.read_csv("train_dataset.csv")
image_folder = "images_resized"

# ==========================================
# 3. TRAIN / VAL / TEST SPLIT
# ==========================================
train_df, temp_df = train_test_split(
    df, test_size=0.3, stratify=df["label_numeric"], random_state=42
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["label_numeric"], random_state=42
)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# ==========================================
# 4. IMAGE LOADER FUNCTION
# ==========================================
def load_images(dataframe):
    X, y = [], []
    
    for _, row in dataframe.iterrows():
        img_path = os.path.join(image_folder, row["image_name"])
        
        img = Image.open(img_path).convert("RGB").resize((224,224))
        img_array = np.array(img)
        img_array = preprocess_input(img_array)  # 🔥 IMPORTANT
        
        X.append(img_array)
        y.append(row["label_numeric"])
    
    return np.array(X), np.array(y)

X_train, y_train = load_images(train_df)
X_val, y_val = load_images(val_df)
X_test, y_test = load_images(test_df)

# ==========================================
# 5. DATA AUGMENTATION
# ==========================================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# ==========================================
# 6. BUILD MODEL (EFFICIENTNET)
# ==========================================
base_model = EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False  # Freeze first

inputs = tf.keras.Input(shape=(224,224,3))
x = data_augmentation(inputs)
x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs, outputs)

# ==========================================
# 7. COMPILE MODEL
# ==========================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

model.summary()

from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# 8. CALLBACKS
# ==========================================
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.3),
]

# ==========================================
# 9. TRAINING (PHASE 1)
# ==========================================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=callbacks
)

# ==========================================
# 10. FINE-TUNING
# ==========================================
base_model.trainable = True

for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-6),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

history_finetune = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50
)

# ==========================================
# 11. FINAL EVALUATION
# ==========================================
results = model.evaluate(X_test, y_test)

print("\n--- Final Test Results ---")
for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.4f}")

# ==========================================
# 12. PREDICTIONS
# ==========================================
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# ==========================================
# 13. CONFUSION MATRIX
# ==========================================
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# ==========================================
# 14. CLASSIFICATION REPORT
# ==========================================
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================================
# 15. ROC CURVE + AUC
# ==========================================
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# ==========================================
# 16. FIND BEST THRESHOLD
# ==========================================
youden_index = tpr - fpr
best_threshold = thresholds[np.argmax(youden_index)]

print(f"\nBest Threshold (Youden Index): {best_threshold:.4f}")

# ==========================================
# 17. SAVE MODEL
# ==========================================
model.save("glaucoma_efficientnet_model.h5")
print("\nModel saved successfully!")

# ==========================================
# 18. TRAINING PLOT
# ==========================================
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()