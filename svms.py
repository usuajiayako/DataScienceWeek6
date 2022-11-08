# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics, svm


# Prepare the data
data = pd.read_csv('fruit_types.csv')
X = data.iloc[:,2:5]
Y = data.iloc[:,0]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=99)

#Create a SVM Classifier
clfLinear = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clfLinear.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clfLinear.predict(X_test)

#Calculate the accuracy of our model
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))