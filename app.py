import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Decision Tree App")

st.title("🌳 Decision Tree Prediction")

@st.cache_data
def load_data():
    return pd.read_csv("winequality-red.csv")

df = load_data()

le = LabelEncoder()
df.iloc[:, -1] = le.fit_transform(df.iloc[:, -1])

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

st.subheader("Enter Input Values")

inputs = []
for col in X.columns:
    inputs.append(
        st.number_input(
            col,
            float(X[col].min()),
            float(X[col].max())
        )
    )

if st.button("Predict"):
    pred = model.predict([inputs])[0]
    st.success(f"Prediction: {le.inverse_transform([pred])[0]}")
