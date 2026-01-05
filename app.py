import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Wine Quality Prediction", layout="centered")

st.title("🍷 Wine Quality Prediction App")
st.write("Predict whether the wine is **Good** or **Bad** using Random Forest")

st.markdown("---")

# Input fields
fixed_acidity = st.number_input("Fixed Acidity", 0.0, 20.0, 7.0)
volatile_acidity = st.number_input("Volatile Acidity", 0.0, 2.0, 0.5)
citric_acid = st.number_input("Citric Acid", 0.0, 1.0, 0.3)
residual_sugar = st.number_input("Residual Sugar", 0.0, 20.0, 2.0)
chlorides = st.number_input("Chlorides", 0.0, 1.0, 0.08)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", 0.0, 100.0, 15.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", 0.0, 300.0, 50.0)
density = st.number_input("Density", 0.9900, 1.0050, 0.9968)
pH = st.number_input("pH", 2.5, 4.5, 3.3)
sulphates = st.number_input("Sulphates", 0.0, 2.0, 0.6)
alcohol = st.number_input("Alcohol (%)", 8.0, 15.0, 10.0)

# Predict button
if st.button("Predict Quality"):
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                             residual_sugar, chlorides, free_sulfur_dioxide,
                             total_sulfur_dioxide, density, pH,
                             sulphates, alcohol]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Good Quality Wine 🍾")
    else:
        st.error("❌ Bad Quality Wine")

st.markdown("---")
st.caption("Model: Random Forest Classifier | Built with Streamlit")
