import pandas as pd  
import streamlit as st  
import altair as alt  

# Set page title  
st.title('Bank Marketing Data Exploration')  

# Load data  
df = pd.read_csv('bank-additional-full.csv', delimiter=';')  # Adjust delimiter if necessary based on the dataset format  

# Project Details  
st.subheader("Abstract")  
st.markdown("This dashboard explores the results of marketing campaigns for a bank.")  
st.markdown("📊 [Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)")  
st.markdown("🐙 [GitHub Repository](https://github.com/Zeraphim/Streamlit-Bank-Marketing-Dashboard)")  
st.markdown("by: [`Zeraphim`](https://jcdiamante.com)")  

# Navigation buttons  
if st.button("Data Cleaning / Pre-processing", use_container_width=True):  
    st.session_state.page_selection = "data_cleaning"  

if st.button("Machine Learning", use_container_width=True):   
    st.session_state.page_selection = "machine_learning"  

if st.button("Prediction", use_container_width=True):   
    st.session_state.page_selection = "prediction"  

if st.button("Conclusion", use_container_width=True):  
    st.session_state.page_selection = "conclusion"  

# Display Data  
st.header('Pré-analyse visuelles données des banques')  

# Description des données  
st.subheader('Description des données')  
if st.checkbox("Show DataFrame"):  
    if st.button("Head"):  
        st.write(df.head(2))  
    if st.button("Tail"):  
        st.write(df.tail())  
    if st.button("Infos"):  
        st.write(df.info())  
    if st.button("Shape"):  
        st.write(df.shape)  

# Create visualizations  
st.subheader('Graphiques de Données')  

# Example visualization: Distribution of the target variable  
target_chart = alt.Chart(df).mark_bar().encode(  
    alt.X('y', title='Target Variable (Success/Failure)'),  
    alt.Y('count()', title='Count'),  
).properties(title="Distribution of Campaign Outcomes")  

st.write(target_chart)  

# Example visualization: Relationship between age and job  
age_job_chart = alt.Chart(df).mark_circle(size=60).encode(  
    x='age',  
    y='job',  
    color='y',  # Target variable  
    tooltip=['age', 'job', 'y']  
).interactive().properties(title="Age vs Job Distribution")  

st.write(age_job_chart)  

# Interactive Design Representation  
st.image("path/to/image.png", caption="Logo de l'application")  

# About  
if st.button("About App"):  
    st.subheader("App d'exploration des données des banques")  
    st.text("Construit avec Streamlit")  
    st.text("Thanks to the Streamlit Team for Amazing Work")  

if st.checkbox("Created By"):  
    st.text("Stéphane C. K. Tékouabou")  
    st.text("junior.kenfack@saintjeanmanagement.org")
