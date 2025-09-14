import streamlit as st
import numpy as np
import json
import re
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import google.generativeai as genai

# —— CONFIG ——  
GOOGLE_API_KEY = ""  
genai.configure(api_key=GOOGLE_API_KEY)  
MODEL_NAME = "gemini-2.0-flash"

# —— LOAD MODEL ——  
model = load_model(
    "best_model.keras",
    custom_objects={"preprocess_input": preprocess_input}
)

class_names = [
    'Apple_Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)Powdery_mildew', 'Cherry(including_sour)_healthy',
    'Corn_(maize)Cercospora_leaf_spot Gray_leaf_spot', 'Corn(maize)_Common_rust',
    'Corn_(maize)Northern_Leaf_Blight', 'Corn(maize)_healthy',
    'Grape_Black_rot', 'Grape_Esca(Black_Measles)', 'Grape_Leaf_blight(Isariopsis_Leaf_Spot)',
    'Grape__healthy', 'Orange_Haunglongbing(Citrus_greening)',
    'Peach_Bacterial_spot', 'Peach_healthy',
    'Pepper,bell_Bacterial_spot', 'Pepper,bell_healthy',
    'Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy',
    'Raspberry_healthy', 'Soybean_healthy',
    'Squash___Powdery_mildew',
    'Strawberry_Leaf_scorch', 'Strawberry_healthy',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites Two-spotted_spider_mite',
    'Tomato_Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Tomato_mosaic_virus',
    'Tomato___healthy'
]

st.title("Plant Disease Detection")
st.write("Upload a leaf image to detect its health status and get treatment info")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
if not uploaded_file:
    st.stop()

# Display and preprocess
img = Image.open(uploaded_file).convert("RGB")
st.image(img, caption="Uploaded Image", use_container_width=True)
img = img.resize((224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Predict
preds = model.predict(img_array)
predicted_index = int(np.argmax(preds))
predicted_class = class_names[predicted_index]
confidence = float(np.max(preds))

# Show basic results
st.markdown(f"### Predicted Class: {predicted_class}")
status = "Healthy" if predicted_class.lower().endswith("healthy") else "Diseased"
st.markdown(f"### Health Status: {status}")
st.markdown(f"### Confidence: {confidence:.1%}")

# Prepare prompt for Gemini
prompt = (
    f"Provide the following information for the plant disease “{predicted_class}” "
    "in strict JSON format with these keys:\n"
    "1. organic_name\n"
    "2. chemical_solution\n"
    "3. preventive_measures\n"
    "4. next_steps\n"
    "For each key, return a string or list of strings as appropriate.\n\n"
    "IMPORTANT: Respond with only the JSON object — no explanations, no labels, "
    "no extra text before or after."
)

# Call Gemini-2.0 Flash via Google Generative AI API
with st.spinner("Fetching treatment information…"):
    model = genai.GenerativeModel(model_name=MODEL_NAME)
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
    except Exception as e:
        st.error(f"Error fetching response from Gemini: {e}")
        st.stop()

# Parse JSON out of response
# ────────────────────────────────────────────────────────────────
# 1) Remove any  fences
# Parse JSON out of response
# ────────────────────────────────────────────────────────────────
clean = text.strip()

# Remove markdown code block fences if present
if clean.startswith(''):
    lines = clean.split('\n')
    # Remove first line (json or )
    lines = lines[1:]
    # Remove last line if it's 
    if lines and lines[-1].strip() == '```':
        lines = lines[:-1]
    clean = '\n'.join(lines)

# Remove any leading "json" label
clean = re.sub(r'^\s*json\s*\n', '', clean, flags=re.IGNORECASE)

# Find the first { and last } to extract just the JSON object
start_idx = clean.find('{')
end_idx = clean.rfind('}')

if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
    clean = clean[start_idx:end_idx+1]

try:
    data = json.loads(clean)
except json.JSONDecodeError:
    st.error("Failed to parse response from Gemini. Here's what it returned:")
    st.code(text)
    st.stop()
# ────────────────────────────────────────────────────────────────

st.header("Treatment & Management Advice")

# Helper to format sections nicely
def display_section(title, content):
    st.subheader(f"🔹 {title}")
    if isinstance(content, list):
        for item in content:
            st.markdown(f"- {item}")
    else:
        st.markdown(content)

if data.get("organic_name"):
    display_section("Organic / Common Treatments", data["organic_name"])

if data.get("chemical_solution"):
    display_section("Chemical Solutions", data["chemical_solution"])

if data.get("preventive_measures"):
    display_section("Preventive Measures", data["preventive_measures"])

if data.get("next_steps"):

    display_section("Recommended Next Steps", data["next_steps"])
