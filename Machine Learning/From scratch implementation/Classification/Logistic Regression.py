# Import pre-processing libs
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import model
from sklearn.linear_model import LogisticRegression

# Import post-processing libs
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
import pickle


###################### 1- Import Data ######################
filename = "Social_Network_Ads.csv"
dataset = pd.read_csv(filename) # Check file extension before using this function
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values


###################### 2- Preprocessing ######################
# preprocess


# Split data
test_train_ratio = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_train_ratio)

# Scale data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

###################### 3- Training ######################


class logistic_regression:
    def __init__(self):
        self.weights = np.array([])
        self.bias = 0
        self.loss = 0
    

    def sigmoid(self, z):    
        output = 1 / (1 + np.exp(-z))
        return output

    def train(self, x, y, learning_rate, iterations): 
        size = x.shape[0]
        weight = np.zeros(x.shape[1])
        bias = 0
        for i in range(iterations): 
            sigma = self.sigmoid(np.dot(x, weight) + bias)
            loss = -1/size * np.sum(y * np.log(sigma)) + (1 - y) * np.log(1-sigma)
            dW = 1/size * np.dot(x.T, (sigma - y))
            db = 1/size * np.sum(sigma - y)
            weight -= learning_rate * dW
            bias -= learning_rate * db 
        
        self.weights = weight
        self.bias = bias
        self.loss = loss
        return self

    def predict(self, x):
        predictions = np.dot(x, self.weights) + self.bias
        return self.sigmoid(predictions) >= 0.5
    
    def score(self, x, y):
        if x.shape[0] == y.shape[0]:
            y_pred = self.predict(x)
            return np.count_nonzero( (y == y_pred) ) / y.shape[0]
        else:
            print("X and y must have the same number of samples")

model = logistic_regression()
model.train(X_train, y_train, learning_rate = 0.02, iterations = 500)


###################### 4- Testing ######################

#model_score = model.score(X_test, y_test)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
score = model.score(X_test, y_test)


###################### 5- Visualization ######################
# Visualising the Training set results
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('yellow', 'black'))(i), label = j)
plt.title('Model fitting (Training set)')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('yellow', 'black'))(i), label = j)
plt.title('Model fitting (Test set)')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()


###################### 6- Save & Use ######################
values_to_predict = X_test
prediction_result = model.predict( values_to_predict )

with open('classifier.pkl', 'wb') as f:
    pickle.dump(model, f)