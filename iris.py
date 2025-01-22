import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

try:
    df = pd.read_csv("data/iris.csv", sep=';', encoding='utf-8') # Correction ici : sep=';'
    # ... (le reste de votre code Streamlit)
except FileNotFoundError:
    st.error("Fichier iris.csv non trouvé. Veuillez vérifier le chemin.")
except pd.errors.ParserError as e:
    st.error(f"Erreur lors de la lecture du fichier CSV : {e}")
except Exception as e:
    st.error(f"Une erreur est survenue : {e}")
