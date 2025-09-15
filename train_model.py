import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import os
import numpy as np
import json

# Environment setup
print("🌱 Starting Plant Disease Detection Training")
print("="*60)

# Check GPU availability
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print(f"🚀 GPU Available: {len(gpus)} device(s)")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("💻 Training on CPU")

# 📂 Dataset paths
train_dir = "dataset/train"
val_dir = "dataset/val"

# Check dataset folders
if not os.path.exists(train_dir):
    print(f"❌ Training folder not found: {train_dir}")
    print("💡 Run clean_and_split.py first!")
    exit()

if not os.path.exists(val_dir):
    print(f"❌ Validation folder not found: {val_dir}")
    print("💡 Run clean_and_split.py first!")
    exit()

# Create models folder
os.makedirs("models", exist_ok=True)

# ✅ Training parameters
IMG_WIDTH = 224
IMG_HEIGHT = 224
BATCH_SIZE = 16
EPOCHS = 10

print(f"📐 Image size: {IMG_WIDTH}x{IMG_HEIGHT}")
print(f"📦 Batch size: {BATCH_SIZE}")
print(f"🔄 Epochs: {EPOCHS}")

# ✅ Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2]
)

val_datagen = ImageDataGenerator(rescale=1./255)

print("📊 Preparing data...")

# Load training data
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

# Load validation data
val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# Dataset info
num_classes = train_data.num_classes
num_train_samples = train_data.samples
num_val_samples = val_data.samples

print(f"📊 Number of classes: {num_classes}")
print(f"📈 Training images: {num_train_samples}")
print(f"📉 Validation images: {num_val_samples}")

# Print class names
print("\n📋 Detected classes:")
class_names = list(train_data.class_indices.keys())
for i, class_name in enumerate(class_names[:10]):
    print(f"  {i:2d}. {class_name}")
if len(class_names) > 10:
    print(f"  ... and {len(class_names)-10} more classes")

# Save class names
with open('models/class_names.txt', 'w', encoding='utf-8') as f:
    for class_name in class_names:
        f.write(f"{class_name}\n")

# ✅ Build model
print("\n🏗️ Building model...")

model = Sequential([
    # Block 1
    Conv2D(32, (3,3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    BatchNormalization(),
    Conv2D(32, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Dropout(0.25),

    # Block 2
    Conv2D(64, (3,3), activation="relu"),
    BatchNormalization(),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Dropout(0.25),

    # Block 3
    Conv2D(128, (3,3), activation="relu"),
    BatchNormalization(),
    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Dropout(0.25),

    # Block 4
    Conv2D(256, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    # Classifier
    Flatten(),
    Dense(512, activation="relu"),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Model summary
print("\n📊 Model summary:")
model.summary()

# Steps
steps_per_epoch = num_train_samples // BATCH_SIZE
validation_steps = num_val_samples // BATCH_SIZE

print(f"\n⚙️ Training setup:")
print(f"📈 Steps per epoch: {steps_per_epoch}")
print(f"📊 Validation steps: {validation_steps}")

# ✅ Callbacks
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001, verbose=1),
    ModelCheckpoint('models/best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
]

print("\n🚀 Training started...")
print("⏰ This may take a while depending on your hardware...")

# Training
try:
    history = model.fit(
        train_data,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_data,
        validation_steps=validation_steps,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    print("\n✅ Training completed successfully!")

except KeyboardInterrupt:
    print("\n⚠️ Training interrupted by user")
    save_anyway = input("Do you want to save the current model? (y/n): ")
    if save_anyway.lower() == 'y':
        model.save("models/interrupted_model.h5")
        print("💾 Temporary model saved")
    exit()

except Exception as e:
    print(f"\n❌ Error during training: {e}")
    exit()

# Save final model
model.save("models/plant_disease_cnn.h5")
print("💾 Model saved at: models/plant_disease_cnn.h5")

# ✅ Plot training history
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
    print("📊 Training curves saved at: models/training_history.png")
    
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    print(f"\n📈 Final results:")
    print(f"🎯 Training accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
    print(f"🎯 Validation accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
    print(f"📉 Training loss: {final_train_loss:.4f}")
    print(f"📉 Validation loss: {final_val_loss:.4f}")
    
    if final_train_acc - final_val_acc > 0.1:
        print("⚠️ Warning: Possible overfitting detected")
        print("💡 Suggestion: Increase Dropout or reduce model complexity")

plot_training_history(history)

# Evaluate final model
print(f"\n📊 Evaluating final model...")
val_loss, val_accuracy = model.evaluate(val_data, verbose=1)
print(f"🎯 Final validation accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")

# Save training info
training_info = {
    'num_classes': num_classes,
    'num_train_samples': num_train_samples,
    'num_val_samples': num_val_samples,
    'img_width': IMG_WIDTH,
    'img_height': IMG_HEIGHT,
    'batch_size': BATCH_SIZE,
    'epochs': len(history.history['accuracy']),
    'final_val_accuracy': float(val_accuracy),
    'final_val_loss': float(val_loss),
    'class_names': class_names
}

with open('models/training_info.json', 'w', encoding='utf-8') as f:
    json.dump(training_info, f, indent=2, ensure_ascii=False)

print(f"\n🎉 Training completed!")
print(f"📁 Saved files:")
print(f"  🤖 models/plant_disease_cnn.h5 (Main model)")
print(f"  🥇 models/best_model.h5 (Best model)")
print(f"  📝 models/class_names.txt (Class names)")
print(f"  📊 models/training_history.png (Training curves)")
print(f"  📋 models/training_info.json (Training info)")

print(f"\n🚀 Next step:")
print(f"python app.py")
print(f"Then open: http://localhost:5000")

print(f"\n💡 Improvement tips:")
if val_accuracy < 0.8:
    print(f"- Try increasing epochs (currently {len(history.history['accuracy'])})")
    print(f"- Try using a pre-trained model (Transfer Learning)")
elif val_accuracy > 0.95:
    print(f"- Excellent model! 🎉")
else:
    print(f"- Good model, can be improved with fine-tuning")

print(f"- Test the model with new images in the app")
print(f"- Monitor performance: check models/training_history.png")
