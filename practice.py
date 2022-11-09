import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('data/diabetes.csv')

#Headings for our data application
st.title("Diabetes Check Data Application")
st.sidebar.header("Patient Data")
st.subheader("Statistical Picture of the Data")
st.write(df.describe())

# Setting me department and independent variable
X = df.drop(["Outcome"], axis = 1)
y = df.iloc[:, -1]

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

def user_report():
    pregnancies = st.sidebar.slider("Pregnancies", 0, 17, 4)
    glucose = st.sidebar.slider("Glucose", 0, 200, 120)
    bp = st.sidebar.slider("BloodPressure", 0, 122, 70)
    skinthickness = st.sidebar.slider("SkinThickness", 0, 100, 20)
    insulin = st.sidebar.slider("Insulin", 0, 846, 80)
    bmi = st.sidebar.slider("BMI", 0, 67, 21)
    dpf = st.sidebar.slider("DiabetesPedigreeFunction", 0.0, 2.4, 0.47)
    age = st.sidebar.slider("Age", 21, 88, 33)
    user_report_data = {
        "Pregnancies": pregnancies,
        "Glucose" : glucose,
        "BloodPressure" : bp,
        "SkinThickness" : skinthickness,
        "Insulin" : insulin,
        "BMI" : bmi,
        "DiabetesPedigreeFunction" : dpf,
        "Age" : age
    }
    report_data = pd.DataFrame(user_report_data, index = [0])
    return report_data

user_data = user_report()
st.header("Patient Data")
st.write(user_data)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
user_result = rf.predict(user_data)