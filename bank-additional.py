import pandas as pd  
import streamlit as st  
import altair as alt  
from sklearn.model_selection import train_test_split  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.preprocessing import LabelEncoder  

# Set page title  
st.title('Bank Marketing Data Exploration')   

# Load data  
df = pd.read_csv('bank-additional-full.csv', delimiter=';')  

# Project Details  
st.subheader("Abstract")  
st.markdown("This dashboard explores the results of marketing campaigns for a bank.")  
st.markdown("📊 [Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)")  
st.markdown("🐙 [GitHub Repository](https://github.com/Zeraphim/Streamlit-Bank-Marketing-Dashboard)")  
st.markdown("by: [`Zeraphim`](https://jcdiamante.com)")  

# Navigation buttons  
menu_options = ['Data Cleaning / Pre-processing', 'Machine Learning', 'Prediction', 'Conclusion']  
page_selection = st.sidebar.selectbox("Select a page", menu_options)  

# Display Data  
st.header('Pré-analyse visuelles données des banques')  

# Description des données  
st.subheader('Description des données')  
if st.checkbox("Show DataFrame"):  
    st.write(df)  

# Visualizations  
st.subheader('Graphiques de Données')  

# Distribution of the target variable  
target_chart = alt.Chart(df).mark_bar().encode(  
    alt.X('y', title='Target Variable (Success/Failure)'),  
    alt.Y('count()', title='Count'),  
).properties(title="Distribution of Campaign Outcomes")  
st.write(target_chart)  

# Relationship between age and job  
age_job_chart = alt.Chart(df).mark_circle(size=60).encode(  
    x='age',  
    y='job',  
    color='y',  # Target variable  
    tooltip=['age', 'job', 'y']  
).interactive().properties(title="Age vs Job Distribution")  
st.write(age_job_chart)  

# Prediction Page  
if page_selection == "Prediction":  
    st.subheader("Make a Prediction")  
    
    # User input fields for prediction  
    age = st.number_input("Age", min_value=18, max_value=100, value=30)  
    job = st.selectbox("Job", df['job'].unique())  
    marital = st.selectbox("Marital Status", df['marital'].unique())  
    education = st.selectbox("Education", df['education'].unique())  
    default = st.selectbox("Credit Default", df['default'].unique())  
    balance = st.number_input("Average Monthly Balance", min_value=-1000, max_value=100000, value=0)  
    housing = st.selectbox("Housing Loan", df['housing'].unique())  
    loan = st.selectbox("Personal Loan", df['loan'].unique())  
    contact = st.selectbox("Contact Communication Type", df['contact'].unique())  
    day = st.number_input("Last Contact Day of Month", min_value=1, max_value=31, value=1)  
    month = st.selectbox("Last Contact Month of Year", df['month'].unique())  
    duration = st.number_input("Last Contact Duration (seconds)", min_value=0, value=0)  
    
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
    
    # Bind input to DataFrame for prediction  
    prediction_df = pd.DataFrame([prediction_input])  
    
    # Encode categorical variables  
    le = LabelEncoder()  
    for col in df.select_dtypes(include=['object']).columns:  
        le.fit(df[col])  
        prediction_df[col] = le.transform(prediction_df[col])  
    
    # Train model (using a simple approach for demo)  
    X = df.drop('y', axis=1)  
    y = df['y']  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
    
    model = RandomForestClassifier()  
    model.fit(X_train, y_train)  
    
    # Make prediction  
    if st.button("Predict"):  
        prediction = model.predict(prediction_df)  
        if prediction[0] == 1:  
            st.success("The model predicts that the client will subscribe to the term deposit.")  
        else:  
            st.warning("The model predicts that the client will not subscribe to the term deposit.")  

# About  
if st.button("About App"):  
    st.subheader("App d'exploration des données des banques")  
    st.text("Construit avec Streamlit")  
    st.text("Thanks to the Streamlit Team for Amazing Work")  

if st.checkbox("Created By"):  
    st.text("Stéphane C. K. Tékouabou")  
    st.text("junior.kenfack@saintjeanmanagement.org")
