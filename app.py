import streamlit as st
import numpy as np
import cv2
import requests
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# --- Dice Coefficient and Loss ---
smooth = 1e-15

def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

# --- Dropbox URL for Model ---
url = "https://www.dropbox.com/scl/fi/ci9go9wim9xxf3clukagc/BHAI_MERA_model.h5?rlkey=kzk2z468492l86bx6lfzucixl&e=1&st=9ox5alwh&dl=0"
model_path = "folder/BHAI_MERA_model1.h5"

# Check if model exists locally, if not, download it
def download_model():
    if not os.path.exists(model_path):
        st.write("Model not found locally, downloading from Dropbox...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                f.write(response.content)
            st.success("Model downloaded successfully!")
        else:
            st.error("Failed to download the model. Please try again later.")
            return None
    return model_path

# --- Load Model Function ---
@st.cache_resource
def load_segmentation_model():
    model_file = download_model()
    if model_file:
        model = load_model(model_file, custom_objects={
            'dice_loss': dice_loss,
            'dice_coef': dice_coef
        })
        model.compile(loss=dice_loss, optimizer=Adam(1e-4), metrics=[dice_coef, 'accuracy'])
        return model
    return None

model = load_segmentation_model()

# --- Mask Prediction & Plotting ---
def generate_mask_overlay(image):
    original = image.copy()
    image = cv2.resize(image, (256, 256))
    x = image / 255.0
    x = np.expand_dims(x, axis=0)

    y_pred = model.predict(x, verbose=0)[0]
    y_pred = np.squeeze(y_pred, axis=-1)
    y_pred = (y_pred >= 0.5).astype(np.uint8)

    mask_rgb = np.zeros_like(image)
    mask_rgb[y_pred == 1] = [255, 0, 0]  # Red overlay

    overlay = cv2.addWeighted(image, 1.0, mask_rgb, 0.5, 0)
    return overlay, image

# --- Streamlit Interface ---
st.title("🧠 Image Segmentation App")
st.write("Upload an image and see the predicted mask.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    overlay, resized_img = generate_mask_overlay(image)

    col1, col2 = st.columns(2)

    with col1:
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption='Overlay Image', use_column_width=True)

    with col2:
        st.image(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB), caption='Resized Input Image', use_column_width=True)

else:
    st.write("Please upload an image to segment.")

