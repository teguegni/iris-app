import streamlit as st
import pandas as pd
import requests  # Pour faire des requêtes HTTP
import altair as alt

# Page configuration
st.set_page_config(page_title="Iris Classification", layout="wide")

# Load data
df = pd.read_csv('iris.csv')

# Set page title
st.title('Iris Classification Dashboard')

# Input form for user to enter features for prediction
st.header('Enter the features for prediction')

sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0)

if st.button("Predict"):
    # Prepare the data for the API request
    input_data = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }
    
    # Send a POST request to the API
    response = requests.post("http://127.0.0.1:5000/predict", json=input_data)
    
    if response.status_code == 200:
        prediction = response.json()
        st.success(f"The predicted species is: {prediction['species'][0]}")
    else:
        st.error(f"Error: {response.json()['error']}")

