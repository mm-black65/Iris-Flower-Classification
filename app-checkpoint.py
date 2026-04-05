#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import numpy as np
import joblib


# In[4]:


from pathlib import Path

# Load trained models and scaler from the script directory
base_dir = Path(__file__).resolve().parent
model_files = {
    "KNN": base_dir / "knn_model.pkl",
    "Logistic Regression": base_dir / "logistic_model.pkl",
    "Decision Tree": base_dir / "tree_model.pkl"
}

models = {}
missing_files = []
for name, path in model_files.items():
    if path.exists():
        models[name] = joblib.load(path)
    else:
        missing_files.append(path.name)

scaler_path = base_dir / "scaler.pkl"
if scaler_path.exists():
    scaler = joblib.load(scaler_path)
else:
    scaler = None
    missing_files.append(scaler_path.name)

if missing_files:
    st.error(f"Missing required files: {', '.join(missing_files)}")
    st.stop()

# Class names
class_names = ["setosa", "versicolor", "virginica"]

# Page config
st.set_page_config(page_title="Iris Classifier", page_icon="🌸")

# Title
st.title("🌸 Iris Flower Classifier")
st.write("Predict the species of an Iris flower using a trained Machine Learning model.")

# Sidebar
st.sidebar.header("About")
st.sidebar.write("This app uses a trained model saved using joblib.")

# Input section
st.subheader("Enter Flower Measurements")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.0)
    sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.0)

with col2:
    petal_length = st.slider("Petal Length", 1.0, 7.0, 4.0)
    petal_width = st.slider("Petal Width", 0.1, 2.5, 1.0)

# Prediction
model_choice = st.selectbox(
    "Choose Model",
    ["KNN", "Logistic Regression", "Decision Tree"]
)

if model_choice == "KNN":
    st.write("📊 Model: KNN (Accuracy ~96%)")
elif model_choice == "Logistic Regression":
    st.write("📊 Model: Logistic Regression (Accuracy ~97%)")
else:
    st.write("📊 Model: Decision Tree (Accuracy ~95%)")

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # ✅ Apply scaling AFTER creating input
    input_data = scaler.transform(input_data)

    prediction = models[model_choice].predict(input_data)
    species = class_names[prediction[0]]

    st.subheader("Prediction Result")

    if species == "setosa":
        st.success(f"The flower is **{species.upper()}** 🌼")
    elif species == "versicolor":
        st.info(f"The flower is **{species.upper()}** 🌷")
    else:
        st.warning(f"The flower is **{species.upper()}** 🌸")

    st.write("### Input Summary")
    st.write(f"Sepal Length: {sepal_length}")
    st.write(f"Sepal Width: {sepal_width}")
    st.write(f"Petal Length: {petal_length}")
    st.write(f"Petal Width: {petal_width}")


st.markdown("---")


# In[ ]:




