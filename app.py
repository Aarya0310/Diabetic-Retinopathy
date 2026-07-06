import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import io

NUM_CLASSES = 5
CLASS_NAMES = ["0 - No DR", "1 - Mild", "2 - Moderate", "3 - Severe", "4 - Proliferative"]

# --- MODEL LOADING ---
@st.cache_resource
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import io
import os
import gdown

NUM_CLASSES = 5
CLASS_NAMES = ["0 - No DR", "1 - Mild", "2 - Moderate", "3 - Severe", "4 - Proliferative"]

# --- MODEL LOADING ---
@st.cache_resource
def load_keras_model():
    """
    Downloads and loads the Keras ResNet50 model.
    """
    file_path = "Diabetic-Retinopathy-ResNet50-model.h5"
    file_id = "1pNnNAy_eZxE0GeaS5GfTnYBM2HhUCLP0" # <--- Put your ID here
    url = f"https://drive.google.com/uc?id={file_id}"
    
    # Download the model only if it hasn't been downloaded yet
    if not os.path.exists(file_path):
        with st.spinner("Downloading model... This may take a minute on the first run."):
            gdown.download(url, file_path, quiet=False)
            
    try:
        model = load_model(file_path)
    except Exception as e:
        st.error(f"Error loading Keras model: {e}")
        st.stop()
        
    return model

# ... (keep the rest of your preprocess_image and app interface code below) ...

# --- IMAGE PREPROCESSING ---
def preprocess_image(image_bytes):
    """
    Preprocesses input image for Keras ResNet50.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize((224, 224))
        img_array = np.array(image).astype(np.float32)
        img_array = np.expand_dims(img_array, axis=0)  # batch size 1
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# --- STREAMLIT APP INTERFACE ---
st.set_page_config(page_title="Diabetic Retinopathy Detection", layout="wide")
st.title("👁️ Diabetic Retinopathy Classification")
st.write("Upload a retinal scan image to classify its DR stage.")

# Load the Keras model
model = load_keras_model()

# File uploader
uploaded_file = st.file_uploader("Choose a retina image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    col1, col2 = st.columns(2)

    with col1:
        st.image(image_bytes, caption="Uploaded Image", use_column_width=True)

    preprocessed_img = preprocess_image(image_bytes)

    if preprocessed_img is not None:
        try:
            # Model predicts probabilities for each class
            probabilities = model.predict(preprocessed_img)[0]
            predicted_index = np.argmax(probabilities)
            predicted_class_name = CLASS_NAMES[predicted_index]

            with col2:
                st.subheader("Prediction Result")
                st.success(f"**Diagnosis:** {predicted_class_name}")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
