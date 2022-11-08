import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = pd.read_csv('fruit_types.csv')  # load our data

encoder = preprocessing.LabelEncoder() # creating labelEncoder

# Converting strings into numbers
fruit_name_encoded=encoder.fit_transform(data.iloc[:,0])
fruit_subtype_encoded=encoder.fit_transform(data.iloc[:,1])
mass_encoded=encoder.fit_transform(data.iloc[:,2])
width_encoded=encoder.fit_transform(data.iloc[:,3])
height_encoded=encoder.fit_transform(data.iloc[:,4])

# Combining encoded data
data=list(zip(fruit_name_encoded,fruit_subtype_encoded,mass_encoded,width_encoded, height_encoded))
data = pd.DataFrame(data, columns = ['fruit_name', 'fruit_subtype', 'mass', 'width', 'height'])
data_to_use = data.iloc[:,1:4]
data_to_target = data.iloc[:,0]

X_train, X_test, y_train, y_test = train_test_split(data_to_use, data_to_target, test_size=0.3, random_state=123) # 70% training, 30% test

# Create default Decision Tree classifer object
decisionTree_model_gini = DecisionTreeClassifier()

# Use entropy criterion
decisionTree_model_entropy = DecisionTreeClassifier(criterion="entropy")

# Use max depth of 3
decisionTree_model_depth3 = DecisionTreeClassifier(max_depth=3)

# Fit the trees
decisionTree_model_gini.fit(X_train,y_train)
decisionTree_model_entropy.fit(X_train,y_train)
decisionTree_model_depth3.fit(X_train,y_train)

# Make prediction
y_pred_gini = decisionTree_model_gini.predict(X_test)
y_pred_entropy = decisionTree_model_entropy.predict(X_test)
y_pred_depth3 = decisionTree_model_depth3.predict(X_test)

print("Accuracy (gini):",metrics.accuracy_score(y_test, y_pred_gini))
print("Accuracy (entropy):",metrics.accuracy_score(y_test, y_pred_entropy))
print("Accuracy (depth3):",metrics.accuracy_score(y_test, y_pred_depth3))






