import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix, confusion_matrix

def main():
    st.title("Titanic Logistic Regression Analysis")
    st.sidebar.title("Display")

main()

#loading data
train_data = pd.read_csv("train.csv")

def original_data():
    st.subheader("Original Test Data")
    st.write(train_data)

#Preprocessing data
train_data["Age"] = train_data["Age"].apply(np.floor)
train_data["Age"].fillna(train_data["Age"].median(skipna= True), inplace = True)
train_data["Sex"] = train_data["Sex"].map({"male": 0, "female": 1})
train_data["Embarked"] = train_data["Embarked"].fillna("S")
# One Hot Encoding for Embarked
encoder = OneHotEncoder()
# Making encoder_df
encoder_df = pd.DataFrame(encoder.fit_transform(train_data[["Embarked"]]).toarray())
encoder_df.columns = ["C", "Q", "S"]
# Join the encoder_df
train_data = train_data.join(encoder_df)
# Drop Embarked column
train_data.drop("Embarked", axis = 1, inplace = True)
# move around columns
train_new_columns = ["PassengerId", "Name", "Ticket", "Cabin", "Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "C", "Q", "S"]
train_data = train_data[train_new_columns]

# spliting train/test data
X = train_data.iloc[:, 5:]
y = train_data.iloc[:, 4]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

scaler_X = StandardScaler() 
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

logisticRegression_model = LogisticRegression(random_state = 10)
logisticRegression_model.fit(X_train, y_train)
prediction = logisticRegression_model.predict(X_test)




# Showing accuracy and confusion_matrix
def cm():
    cm = confusion_matrix(y_test, prediction)
    plot_confusion_matrix(logisticRegression_model, X_test, y_test, display_labels=["Deceased", "Suvived"])
    st.subheader("Confusion Matrix")
    confusion_mx = Figure()
    ax = confusion_mx.subplots()
    plot_confusion_matrix(logisticRegression_model, X_test, y_test, display_labels=["Deceased", "Suvived"], ax = ax)
    st.pyplot(confusion_mx)
    st.write("Accuracy: ", (cm[0][0] + cm[1][1])/ len(y_test))

def sex_survived():
    st.subheader("Sex vs Survived")
    sex_vs_survived_fig = Figure()
    ax = sex_vs_survived_fig.subplots()
    sns.regplot(x = X["Sex"], y = y, logistic = True, ax = ax)
    st.pyplot(sex_vs_survived_fig)

def age_survived():
    st.subheader("Age vs Survived")
    age_vs_survived_fig = Figure()
    ax = age_vs_survived_fig.subplots()
    sns.regplot(x = X["Age"], y = y, logistic = True, ax = ax)
    st.pyplot(age_vs_survived_fig)

def fare_survived():
    st.subheader("Fare vs Survived")
    fare_vs_survived_fig = Figure()
    ax = fare_vs_survived_fig.subplots()
    sns.regplot(x = X["Fare"], y = y, logistic = True, ax = ax)
    st.pyplot(fare_vs_survived_fig)

if st.sidebar.checkbox("Original Data", False):
    original_data()

if st.sidebar.checkbox("Confusion Matrix", False):
    cm()

if st.sidebar.checkbox("Sex vs Survived", False):
    sex_survived()

if st.sidebar.checkbox("Age vs Survived", False):
    age_survived()

if st.sidebar.checkbox("Fare vs Survived", False):
    fare_survived()