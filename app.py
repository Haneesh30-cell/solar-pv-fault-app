import streamlit as st
import joblib
import numpy as np

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Intelligent Solar PV Fault Prediction",
    layout="centered"
)

st.title("Intelligent Solar PV Fault Prediction")
st.markdown(
    "Enter 18 extracted electrical feature values (comma separated) "
    "to predict the PV array operating condition."
)

# ------------------------------
# Load Model
# ------------------------------
@st.cache_resource
def load_model():
    return joblib.load("pv_model.pkl")

model = load_model()

# ------------------------------
# Class Labels (IMPORTANT)
# Order must match training order
# ------------------------------
class_labels = [
    "Healthy",
    "Open Circuit Fault",
    "Short Circuit Fault",
    "Temperature Variation",
    "Partial Shading",
    "Bypass Diode Fault"
]

# ------------------------------
# Input Box
# ------------------------------
features_input = st.text_area(
    "Enter 18 feature values (comma separated)",
    height=150,
    placeholder="Example:\n0.12, 5.34, 100.5, ..."
)

# ------------------------------
# Prediction Button
# ------------------------------
if st.button("Predict Fault"):

    try:
        # Convert input string to float list
        values = [float(x.strip()) for x in features_input.split(",") if x.strip() != ""]

        # Validate length
        if len(values) != 18:
            st.error(f"Please enter exactly 18 values. You entered {len(values)}.")
        else:
            # Convert to numpy array
            data = np.array(values).reshape(1, -1)

            # Get raw prediction
            raw_prediction = model.predict(data)

            # If ANN (softmax output)
            if len(raw_prediction[0]) > 1:
                class_index = np.argmax(raw_prediction[0])
                confidence = round(float(np.max(raw_prediction[0])) * 100, 2)
            else:
                class_index = int(raw_prediction[0])
                confidence = None

            prediction = class_labels[class_index]

            # Display result
            st.markdown("---")

            if prediction.lower() == "healthy":
                st.success(f"System Status: HEALTHY")
            else:
                st.error(f"Fault Detected: {prediction}")

            if confidence is not None:
                st.info(f"Prediction Confidence: {confidence}%")

    except:
        st.error("Invalid input format. Please enter numeric values separated by commas.")
