# Importer les bibliothèques nécessaires
import streamlit as st
import pandas as pd
import altair as alt

# Configuration de la page
st.set_page_config(
    page_title="Classification des Iris", 
    page_icon="assets/icon/icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

alt.themes.enable("dark")

# -------------------------
# Barre latérale

# Initialiser page_selection dans l'état de session si pas déjà défini
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'a_propos'  # Page par défaut
pass
# Fonction pour mettre à jour page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

with st.sidebar:
    st.title('Classification des Iris')
    
    # Navigation par boutons avec des icônes
    st.subheader("Sections")
    
    if st.button("🏷️ À Propos", use_container_width=True, on_click=set_page_selection, args=('a_propos',)):
        st.session_state.page_selection = 'a_propos'
    pass
    if st.button("📊 Jeu de Données", use_container_width=True, on_click=set_page_selection, args=('jeu_de_donnees',)):
        st.session_state.page_selection = 'jeu_de_donnees'
pass
    if st.button("🔍 Analyse Exploratoire", use_container_width=True, on_click=set_page_selection, args=('analyse_exploratoire',)):
        st.session_state.page_selection = "analyse_exploratoire"
pass
    if st.button("🧹 Nettoyage / Prétraitement", use_container_width=True, on_click=set_page_selection, args=('nettoyage_donnees',)):
        st.session_state.page_selection = "nettoyage_donnees"
pass
    if st.button("🤖 Apprentissage Automatique", use_container_width=True, on_click=set_page_selection, args=('apprentissage_automatique',)): 
        st.session_state.page_selection = "apprentissage_automatique"
pass
    if st.button("🔮 Prédiction", use_container_width=True, on_click=set_page_selection, args=('prediction',)): 
        st.session_state.page_selection = "prediction"
pass 
    if st.button("📜 Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        st.session_state.page_selection = "conclusion"
pass
    # Détails du projet
    st.subheader("Résumé")
    st.markdown("Un tableau de bord Streamlit mettant en évidence les résultats de l'entraînement de deux modèles de classification utilisant le jeu de données des fleurs Iris.")
    st.markdown("📊 [Jeu de Données](https://www.kaggle.com/datasets/arshid/iris-flower-dataset)")
    st.markdown("📗 [Notebook Google Colab](https://colab.research.google.com/drive/1KJDBrx3akSPUW42Kbeepj64ZisHFD-NV?usp=sharing)")
    st.markdown("🐙 [Dépôt GitHub](https://github.com/Zeraphim/Streamlit-Iris-Classification-Dashboard)")
    st.markdown("par : [`Zeraphim`](https://jcdiamante.com)")

# -------------------------

# Charger les données
try:
    df = pd.read_csv('iris.csv', delimiter=',')
except FileNotFoundError:
    st.error("Le fichier 'iris.csv' est introuvable. Veuillez vérifier son emplacement.")
    st.stop()

# Définir le titre de la page
st.title('ISJM BI - Exploration des données des Iris')

st.header('Pré-analyse visuelle des données des Iris TP1')

# Afficher les premières lignes des données chargées
st.subheader('Description des données')

# Afficher le jeu de données avec des boutons de prévisualisation
if st.checkbox("Afficher les options de prévisualisation du DataFrame"):
    if st.button("Afficher les 2 premières lignes"):
        st.write(df.head(2))
    if st.button("Afficher les dernières lignes"):
        st.write(df.tail())
    if st.button("Afficher les informations sur le DataFrame"):
        buffer = pd.io.formats.style.Styler(df)
        buffer.set_table_attributes('style="width:100%"')
        buffer.set_properties(**{'text-align': 'left'})
        st.write(buffer)
    if st.button("Afficher la forme du DataFrame"):
        st.write(df.shape)
else:
    st.write(df.head(2))

# Créer un graphique avec Altair
chart = alt.Chart(df).mark_point().encode(
    x='petal_length',
    y='petal_width',
    color='species',
).properties(title='Relation entre la longueur et la largeur du pétale')

# Afficher le graphique
st.altair_chart(chart, use_container_width=True)

# Représentation interactive avec Altair
chart2 = alt.Chart(df).mark_circle(size=60).encode(
    x='sepal_length',
    y='sepal_width',
    color='species',
    tooltip=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
).interactive().properties(title='Nuage de points : Longueur vs Largeur du sépale')

st.altair_chart(chart2, use_container_width=True)

# À propos de l'application
if st.button("À Propos de l'Application"):
    st.subheader("Application d'exploration des données des Iris")
    st.text("Construite avec Streamlit")
    st.text("Merci à l'équipe Streamlit pour leur travail incroyable")

if st.checkbox("Créé par"):
    st.text("Stéphane C. K. Tékouabou")
    st.text("ctekouaboukoumetio@gmail.com")
