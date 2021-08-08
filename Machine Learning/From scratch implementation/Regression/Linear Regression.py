# Import pre-processing libs
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import model
from sklearn.linear_model import LinearRegression

# Import post-processing libs
import matplotlib.pyplot as plt
import pickle


###################### 1- Import Data ######################
filename = "linear_reg_DS.csv"
dataset = pd.read_csv(filename) # Check file extension before using this function
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 1:].values


###################### 2- Preprocessing ######################
# preprocess

# Split data
test_train_ratio = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_train_ratio)


###################### 3- Training ######################
class linear_regression:
    def __init__(self):
        self.b_0 = None
        self.b_1 = None
    
    def fit(self, x, y): 
        # Estimate Coef
        # number of observations/points 
        n = np.size(x) 
      
        # mean of x and y vector 
        m_x, m_y = np.mean(x), np.mean(y) 
      
        # calculating cross-deviation and deviation about x 
        SS_xy = np.sum(y*x) - n*m_y*m_x 
        SS_xx = np.sum(x*x) - n*m_x*m_x 
      
        # calculating regression coefficients 
        b_1 = SS_xy / SS_xx 
        b_0 = m_y - b_1*m_x 
      
        self.b_0 = b_0
        self.b_1 = b_1
    
    def predict(self, x):
        y_pred = self.b_0 + self.b_1 * x
        return y_pred
    
    def score(self, X_test, y_test):
        # Initiate parameters :  y | y mean | y predicted
        y = y_test
        y_mean = np.mean(y_test)
        y_pred = self.predict(X_test)
        # Calculate R2
        SSR = np.sum( np.square( y - y_pred ) )
        SST = np.sum( np.square( y - y_mean ) )
        coef_r2_det = 1 - (SSR / SST)
        return coef_r2_det
    
    
model = linear_regression()
model.fit(X_train, y_train)



###################### 4- Testing ######################
model_score = model.score(X_test, y_test)


###################### 5- Visualization ######################
plt.scatter(X_test, y_test, color="red")
plt.plot(X_test, model.predict(X_test), color="blue")
plt.show()


###################### 6- Save & Use ######################
values_to_predict = X_test
prediction_result = model.predict(values_to_predict)

with open('linearModel.pkl', 'wb') as f:
    pickle.dump(model, f)
