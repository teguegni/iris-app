import streamlit as st
import pandas as pd
import os

# Page configuration
st.set_page_config(page_title="Iris Classification", layout="wide")

# Load data
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'iris.csv')

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    st.error("Le fichier 'iris.csv' est introuvable. Veuillez vérifier son emplacement.")
    st.stop()
except Exception as e:
    st.error(f"Une erreur s'est produite : {e}")
    st.stop()

# Set page title
st.title('Iris Classification Dashboard')

# Display the dataset
st.header('Dataset Overview')
if st.checkbox("Show Dataset"):
    st.write(df)

# Continue with your Streamlit app...


