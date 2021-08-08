# Import pre-processing libs
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math
from collections import defaultdict
# Import model
from sklearn.naive_bayes import GaussianNB  # MultinomialNB

# Import post-processing libs
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
import pickle


###################### 1- Import Data ######################
filename = 'Social_Network_Ads.csv'
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

class NaiveBayes:
    def __init__(self, X, y):
        self.num_examples, self.num_features = X.shape
        self.num_classes = len(np.unique(y))
        self.eps = 1e-6

    def fit(self, X, y):
        self.classes_mean = {}
        self.classes_variance = {}
        self.classes_prior = {}

        for c in range(self.num_classes):
            X_c = X[y == c]

            self.classes_mean[str(c)] = np.mean(X_c, axis=0)
            self.classes_variance[str(c)] = np.var(X_c, axis=0)
            self.classes_prior[str(c)] = X_c.shape[0] / X.shape[0]

    def predict(self, X):
        probs = np.zeros((X.shape[0], self.num_classes))

        for c in range(self.num_classes):
            prior = self.classes_prior[str(c)]
            probs_c = self.density_function(
                X, self.classes_mean[str(c)], self.classes_variance[str(c)]
            )
            probs[:, c] = probs_c + np.log(prior)

        return np.argmax(probs, 1)

    def density_function(self, x, mean, sigma):
        # Calculate probability from Gaussian density function
        const = -self.num_features / 2 * np.log(2 * np.pi) - 0.5 * np.sum(
            np.log(sigma + self.eps)
        )
        probs = 0.5 * np.sum(np.power(x - mean, 2) / (sigma + self.eps), 1)
        return const - probs
    
    def score(self, y_true, y_pred):
        if y_true.shape[0] == y_pred.shape[0]:
            return np.count_nonzero( (y_pred == y_true) ) / y_true.shape[0]
        else:
            print("y_true and y_pred must have the same number of samples")
    
model = NaiveBayes(X_train, y_train)
model.fit(X_train, y_train)
  
# test model 

###################### 4- Testing ######################

#model_score = model.score(X_test, y_test)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
score = model.score(y_test, y_pred)

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
prediction_result = model.predict([ values_to_predict ])

with open('classifier.pkl', 'wb') as f:
    pickle.dump(model, f)