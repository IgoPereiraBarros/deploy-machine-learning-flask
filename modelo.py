# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import requests
import json

# Importing the dataset
df = pd.read_csv('dataset/Salary_Data.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Splitting dataset in Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# Fitting the model LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Prediting the result
y_pred = regressor.predict(X_test)

# Save the model
pickle.dump(regressor, open('saved_model/model.pkl', 'wb'))

# Prediting the result with other data
model = pickle.load(open('saved_model/model.pkl', 'rb'))
print(model.predict([[1.8]]))
