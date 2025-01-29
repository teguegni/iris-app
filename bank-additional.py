import pandas as pd  
import streamlit as st  
import altair as alt  
from sklearn.model_selection import train_test_split  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.preprocessing import LabelEncoder  

# Set up title for the page  
st.title('Exploration des Données de Marketing Bancaire')   

# Load the data  
df = pd.read_csv('bank-additional-full.csv', delimiter=';')  

# Project details  
st.subheader("Résumé")  
st.markdown("Ce tableau de bord explore les résultats des campagnes de marketing pour une banque.")  
st.markdown("📊 [Jeu de données](https://archive.ics.uci.edu/ml/datasets/bank+marketing)")  
st.markdown("🐙 [Dépôt GitHub](https://github.com/Zeraphim/Streamlit-Bank-Marketing-Dashboard)")  
st.markdown("par : [`Zeraphim`](https://jcdiamante.com)")  

# Navigation buttons  
menu_options = ['Nettoyage / Pré-traitement des données', 'Apprentissage Automatique', 'Prédiction', 'Conclusion']  
page_selection = st.sidebar.selectbox("Sélectionner une page", menu_options)  

# Display data  
st.header('Pré-analyse Visuelle des Données Bancaires')  

# Data description  
st.subheader('Description des données')  
if st.checkbox("Afficher le DataFrame"):  
    st.write(df)  

# Visualizations  
st.subheader('Graphiques de Données')  

# Distribution of target variable  
target_chart = alt.Chart(df).mark_bar().encode(  
    alt.X('y', title='Variable Cible (Succès/Échec)'),  
    alt.Y('count()', title='Nombre'),  
).properties(title="Distribution des Résultats de la Campagne")  
st.write(target_chart)  

# Relationship between age and job  
age_job_chart = alt.Chart(df).mark_circle(size=60).encode(  
    x='age',  
    y='job',  
    color='y',  # Target variable  
    tooltip=['age', 'job', 'y']  
).interactive().properties(title="Distribution de l'Âge par Métier")  
st.write(age_job_chart)  

# Prediction Page Logic
if page_selection == "Prédiction":  
    st.subheader("Faire une Prédiction")  
    
    # User input fields for prediction 
    age = st.number_input("Âge", min_value=18, max_value=100, value=30)  
    job = st.selectbox("Métier", df['job'].unique())  
    marital = st.selectbox("État Civil", df['marital'].unique())  
    education = st.selectbox("Éducation", df['education'].unique())  
    default = st.selectbox("Défaut de Crédit", df['default'].unique())  
    balance = st.number_input("Solde Mensuel Moyen", min_value=-1000, max_value=100000, value=0)  
    housing = st.selectbox("Prêt Logement", df['housing'].unique())  
    loan = st.selectbox("Prêt Personnel", df['loan'].unique())  
    contact = st.selectbox("Type de Communication", df['contact'].unique())  
    day = st.number_input("Jour du Dernier Contact", min_value=1, max_value=31, value=1)  
    month = st.selectbox("Mois du Dernier Contact", df['month'].unique())  
    duration = st.number_input("Durée du Dernier Contact (secondes)", min_value=0, value=0)  
    
    # Prepare input for prediction 
    prediction_input = {  
        "age": age,  
        "job": job,  
        "marital": marital,  
        "education": education,  
        "default": default,  
        "balance": balance,  
        "housing": housing,  
        "loan": loan,  
        "contact": contact,  
        "day": day,  
        "month": month,  
        "duration": duration 
    }  
    
    # Convert input to DataFrame for prediction 
    prediction_df = pd.DataFrame([prediction_input])  
    
    # Encode categorical variables 
    le = LabelEncoder()  
    
    for col in df.select_dtypes(include=['object']).columns: 
        le.fit(df[col]) 
        prediction_df[col] = le.transform(prediction_df[col])  
    
    # Train model only once when needed
    if 'model' not in st.session_state:
        X = df.drop('y', axis=1)  # Features
        y = df['y'].map({'yes': 1, 'no': 0})  # Target variable
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

        model = RandomForestClassifier(random_state=42)  # Initialize model
        model.fit(X_train, y_train)  # Train model
        st.session_state.model = model  # Store model in session state

    # Make a prediction 
    if st.button("Prédire"): 
        prediction = st.session_state.model.predict(prediction_df) 
        if prediction[0] == 1: 
            st.success("Le modèle prédit que le client va s'abonner au dépôt à terme.") 
        else: 
            st.warning("Le modèle prédit que le client ne s'abonnera pas au dépôt à terme.")  

# About section 
if st.button("À propos de l'application"): 
    st.subheader("Application d'Exploration des Données Bancaires") 
    st.text("Construit avec Streamlit") 
    st.text("Merci à l'équipe Streamlit pour leur travail incroyable") 

if st.checkbox("Créé par"): 
    st.text("Stéphane C. K. Tékouabou") 
    st.text("junior.kenfack@saintjeanmanagement.org")

  

   
