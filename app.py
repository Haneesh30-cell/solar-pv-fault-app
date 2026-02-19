import streamlit as st
import joblib
import numpy as np

# Page configuration
st.set_page_config(page_title="Intelligent Solar PV Fault Prediction", layout="centered")

st.title("Intelligent Solar PV Fault Prediction")

st.markdown("Enter 18 extracted electrical feature values (comma separated)")

# Load trained model
model = joblib.load("pv_model.pkl")

# Input box
features = st.text_area(
    "Enter 18 feature values (comma separated)",
    height=150
)

# Prediction button
if st.button("Predict Fault"):

    try:
        # Convert input string to list of floats
        values = [float(x.strip()) for x in features.split(",") if x.strip() != ""]

        # Check length
        if len(values) != 18:
            st.error(f"Please enter exactly 18 values. You entered {len(values)}.")
        else:
            # Convert to numpy array
            data = np.array(values).reshape(1, -1)

            # Predict
            prediction = model.predict(data)[0]

            # Confidence (if model supports probability)
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(data)[0]
                confidence = round(float(np.max(probabilities)) * 100, 2)
            else:
                confidence = None

            # Display result
            if str(prediction).lower() == "healthy":
                st.success("System Status: HEALTHY")
                if confidence:
                    st.write(f"Confidence: {confidence}%")
            else:
                st.error(f"Fault Detected: {prediction}")
                if confidence:
                    st.write(f"Confidence: {confidence}%")

    except Exception as e:
        st.error("Invalid input format. Please enter numeric values separated by commas.")
