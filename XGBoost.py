# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
dataset = pd.read_csv("Churn_Modelling.csv")

#Independent Variable Matrix/ Vector
X = dataset.iloc[:,3:13].values

#Making Dependent Variable Matrix/ Vector
y= dataset.iloc[:, 13].values

#Encoding/Labeling Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
#encoding the coutnry
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
#encoding the genders
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])

transformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [1])],remainder='passthrough')
X = np.array(transformer.fit_transform(X), dtype=np.float)

X = X[:, 1:]

#Splitting the dataset into Training and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting XGBoost to the Training Set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

#Predicting Single Value/ new result with regression
y_pred = classifier.predict(X_test)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Applying K Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv=10)
accuracies.mean()
accuracies.std()
