import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "plant_disease_cnn.h5")

model = tf.keras.models.load_model(MODEL_PATH)
# أسماء الكلاسات (نفس اللي في التدريب)
class_labels = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
    'Apple___healthy', 'Blueberry___healthy', 'Cherry___Powdery_mildew',
    'Cherry___healthy', 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy',
    # ✨ كمل باقي الكلاسات من train_data.class_indices.keys()
]

def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    return class_labels[class_idx], float(confidence)
