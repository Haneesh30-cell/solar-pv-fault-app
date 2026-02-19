import streamlit as st
import joblib
import numpy as np

st.title("Intelligent Solar PV Fault Prediction")

model = joblib.load("pv_model.pkl")

features = st.text_area("Enter 19 feature values (comma separated)")

if st.button("Predict Fault"):

    try:
        values = [float(x.strip()) for x in features.split(",")]

        if len(values) != 19:
            st.error("Please enter exactly 19 values.")
        else:
            data = np.array(values).reshape(1, -1)

            prediction = model.predict(data)[0]
            probabilities = model.predict_proba(data)[0]
            confidence = round(float(np.max(probabilities)) * 100, 2)

            if prediction.lower() == "healthy":
                st.success(f"System Healthy\nConfidence: {confidence}%")
            else:
                st.error(f"Fault Detected: {prediction}\nConfidence: {confidence}%")

    except:
        st.error("Invalid input format.")
