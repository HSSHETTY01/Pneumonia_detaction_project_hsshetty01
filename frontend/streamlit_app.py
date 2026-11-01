import io
import requests
import streamlit as st
import numpy as np
import pydicom
from PIL import Image

st.set_page_config(page_title="Pneumonia Detection Assistant", page_icon="🫁", layout="wide")
BACKEND_URL = st.sidebar.text_input("Backend API URL", value="http://backend:5000/predict")
st.title("🫁 Pulmonary Opacity & Pneumonia Detection UI")
uploaded_file = st.file_uploader("Choose a Chest X-ray image...", type=["dcm", "png", "jpg", "jpeg"])
col1, col2 = st.columns([1, 1])

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    with col1:
        st.subheader("Uploaded Radiograph")
        if uploaded_file.name.lower().endswith(".dcm"):
            dcm = pydicom.dcmread(io.BytesIO(file_bytes))
            arr = dcm.pixel_array.astype(np.float32)
            arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-7) * 255.0
            display_img = Image.fromarray(arr.astype(np.uint8))
        else:
            display_img = Image.open(io.BytesIO(file_bytes))
        st.image(display_img, caption=f"Filename: {uploaded_file.name}", use_container_width=True)

    with col2:
        st.subheader("Diagnostic Prediction")
        if st.button("Run Diagnostic Inference", type="primary"):
            with st.spinner("Analyzing..."):
                try:
                    files = {"file": (uploaded_file.name, file_bytes)}
                    response = requests.post(BACKEND_URL, files=files, timeout=60)
                    if response.status_code == 200:
                        result = response.json()
                        pred_class = result["predicted_class"]
                        prob_pneumonia = result["pneumonia_probability"]
                        prob_normal = result["normal_probability"]
                        threshold = result["operating_threshold"]

                        if pred_class == "Lung Opacity / Pneumonia":
                            st.error(f"### Detection: **{pred_class}**")
                        else:
                            st.success(f"### Detection: **{pred_class}**")

                        st.metric(
                            label="Pneumonia Confidence",
                            value=f"{prob_pneumonia:.2%}",
                            delta=f"{(prob_pneumonia - threshold):.2%} vs cutoff"
                        )
                        st.progress(prob_pneumonia)
                    else:
                        st.error(f"Backend error: {response.status_code}")
                except Exception as err:
                    st.error(f"Failed connection: {err}")
