# Import pre-processing libs
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import model
from sklearn.svm import SVR

# Import post-processing libs
import matplotlib.pyplot as plt
import pickle

###################### 1- Import Data ######################
filename = ""
dataset = pd.read_csv(filename) # Check file extension before using this function
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 1:].values

###################### 2- Preprocessing ######################
# preprocess

# Split data
test_train_ratio = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_train_ratio)


###################### 3- Training ######################
model = SVR(kernel = "rbf", C = 1.0, epsilon = 0.1, tol = 0.001, )
model.fit(X_train, y_train)


###################### 4- Testing ######################
model_score = model.score(X_test, y_test)


###################### 5- Visualization ######################
plt.scatter(X_test, y_test, color="red")
plt.plot(X_test, model.predict(X_test), color="blue")
plt.show()


###################### 6- Save & Use ######################
values_to_predict = X_test
prediction_result = model.predict([ values_to_predict ])

with open('regressor.pkl', 'wb') as f:
    pickle.dump(model, f)
