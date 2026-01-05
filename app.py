import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Decision Tree App")

st.title("🌳 Decision Tree Classifier")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("dataset.csv")

df = load_data()

# Encode target
le = LabelEncoder()
df.iloc[:, -1] = le.fit_transform(df.iloc[:, -1])

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Train model INSIDE app
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# User Input
st.subheader("Enter Feature Values")

user_input = []
for col in X.columns:
    value = st.number_input(col, float(X[col].min()), float(X[col].max()))
    user_input.append(value)

# Predict
if st.button("Predict"):
    prediction = model.predict([user_input])[0]
    st.success(f"Prediction: {le.inverse_transform([prediction])[0]}")
