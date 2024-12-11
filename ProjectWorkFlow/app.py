import streamlit as st
import os
import numpy as np
import torch
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from torchvision.transforms import functional as F
from PIL import Image
import pickle

# Suppress warnings
import warnings

warnings.filterwarnings("ignore")

# Function to add a background image using a link
def add_background_image(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
        }}
        .center-title {{
            text-align: center;
            color: white;
            font-size: 50px;
            font-weight: bold;
        }}
        .results-text {{
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Add background image using a URL
add_background_image("https://c4.wallpaperflare.com/wallpaper/954/310/420/fast-and-furious-7-movie-scene-wallpaper-preview.jpg")  # Replace with your image URL

# Load the models
@st.cache_resource
def load_models():
    tf_models = {
        "car_detection": load_model('models/car_classification_model.h5'),
        "brand_detection": load_model('models/brand_detection_model.h5'),
        "damage_detection": load_model('models/damage_detection_model.h5'),
        "location_detection": load_model('models/location_detection_model.h5'),
        "severity": load_model('models/severity_model.h5')
    }
    part_detection_model = torch.load('models/Part_predection_model.pkl', map_location=torch.device('cpu'))
    part_detection_model.eval()
    with open('models/price_P_model.pkl', 'rb') as file:
        price_model = pickle.load(file)
    return tf_models, part_detection_model, price_model

# Load the models
tf_models, part_detection_model, price_model = load_models()

brand_categories = ["Ford_Mustang", "Mazda_3", "Toyota_Camry"]

# Hardcoded label map for part detection
label_map = {
    0: "background", 1: "Damage-Windshield", 2: "Damage-boots", 3: "Damage-door",
    4: "Damage-front-bumper", 5: "Damage-headlight", 6: "Damage-hood",
    7: "Damage-left-fender", 8: "Damage-rear-bumper", 9: "Damage-rear-light",
    10: "Damage-right-fender", 11: "Damage-roof", 12: "Damage-side-mirror",
    13: "Damage-window-glass", 14: "flat-tire"
}

# Helper Functions
def preprocess_image_tensorflow(img_path, img_width, img_height):
    img = image.load_img(img_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def preprocess_image_pytorch(img_path):
    img = Image.open(img_path).convert("RGB")
    img_tensor = F.to_tensor(img)
    return img_tensor, img

# Main detection pipeline
def complete_detection_pipeline(img_path, confidence_threshold=0.5):
    features = {}

    # Step 1: Car classification
    car_input_shape = tf_models['car_detection'].input_shape[1:3]
    car_image = preprocess_image_tensorflow(img_path, *car_input_shape)
    car_prediction = tf_models['car_detection'].predict(car_image)[0][0]
    features['car_detected'] = "Car Detected" if car_prediction >= 0.5 else "No Car Detected"

    if car_prediction >= 0.5:
        # Brand detection
        brand_input_shape = tf_models['brand_detection'].input_shape[1:3]
        brand_image = preprocess_image_tensorflow(img_path, *brand_input_shape)
        brand_prediction = tf_models['brand_detection'].predict(brand_image)
        brand_class = np.argmax(brand_prediction)
        features['brand'] = brand_categories[brand_class]
        

        # Damage detection
        damage_input_shape = tf_models['damage_detection'].input_shape[1:3]
        damage_image = preprocess_image_tensorflow(img_path, *damage_input_shape)
        damage_prediction = tf_models['damage_detection'].predict(damage_image)[0][0]
        features['damage_detected'] = "Damage Detected" if car_prediction >= 0.5 else "No Car Detected"

        # Location detection
        location_input_shape = tf_models['location_detection'].input_shape[1:3]
        location_image = preprocess_image_tensorflow(img_path, *location_input_shape)
        location_prediction = tf_models['location_detection'].predict(location_image)
        location_class = np.argmax(location_prediction)
        location_confidence = location_prediction[0][location_class]
        location_labels = ['front', 'side', 'back']
        features['location'] = location_labels[location_class]

        # Severity check
        severity_input_shape = tf_models['severity'].input_shape[1:3]
        severity_image = preprocess_image_tensorflow(img_path, *severity_input_shape)
        severity_prediction = tf_models['severity'].predict(severity_image)
        severity_class = np.argmax(severity_prediction)
        severity_labels = ['minor', 'moderate', 'severe']
        features['severity'] = severity_labels[severity_class]

        # Part detection
        img_tensor, _ = preprocess_image_pytorch(img_path)
        img_tensor = img_tensor.unsqueeze(0)
        with torch.no_grad():
            outputs = part_detection_model(img_tensor)[0]
            outputs = {k: v for k, v in outputs.items() if len(v) > 0}
            parts_detected = [label_map[label.item()] for i, label in enumerate(outputs['labels']) if
                              outputs['scores'][i].item() >= confidence_threshold]
            features['parts_detected'] = parts_detected

        # Step 7: Price prediction
        input_vector = np.zeros(23)  # Replace 23 with the exact number of features used in training

        # Fill in features (ensure alignment with training data)
        input_vector[0] = car_prediction
        input_vector[1] = 1 if features.get('brand', '') == "Mazda_3" else 0
        input_vector[2] = 1 if features.get('brand', '') == "Ford_Mustang" else 0
        input_vector[3] = 1 if features.get('brand', '') == "Toyota_Camry" else 0
        input_vector[4] = damage_prediction
        input_vector[5] = 1 if features.get('location', '') == "front" else 0
        input_vector[6] = 1 if features.get('location', '') == "side" else 0
        input_vector[7] = 1 if features.get('location', '') == "back" else 0
        input_vector[8] = severity_prediction[0][severity_class]

        # Parts detected (binary indicators)
        for part_index, part_name in label_map.items():
            if part_name in features.get('parts_detected', []):
                input_vector[9 + part_index] = 1

        # Reshape for prediction
        input_vector = input_vector.reshape(1, -1)
        predicted_price = np.expm1(price_model.predict(input_vector)[0])
        features['predicted_price'] = predicted_price

    return features

# Streamlit app
st.markdown("<h1 class='center-title'>CrashCal</h1>", unsafe_allow_html=True)
st.write("<div style='color: white; text-align: center;'>Welcome to CrashCal! Upload your car image to get a detailed report.</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an image of a car", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with st.spinner("Processing the image..."):
        # Save the uploaded file temporarily
        img_path = "uploaded_image.jpg"
        with open(img_path, "wb") as f:
            f.write(uploaded_file.read())

        # Run the pipeline
        features = complete_detection_pipeline(img_path)

        # Display results
        st.image(img_path, caption="Uploaded Image", use_column_width=True)
        st.subheader("Detection Results")
        for key, value in features.items():
            if key == "predicted_price":
                st.markdown(f"<div class='results-text'><b>Estimated Price</b>: ${value:.2f}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='results-text'><b>{key.replace('_', ' ').capitalize()}</b>: {value}</div>", unsafe_allow_html=True)
