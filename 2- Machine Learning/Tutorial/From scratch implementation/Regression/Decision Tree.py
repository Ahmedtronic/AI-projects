# Import pre-processing libs
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import model
from sklearn.tree import DecisionTreeRegressor

# Import post-processing libs
import matplotlib.pyplot as plt
import pickle


###################### 1- Import Data ######################
filename = "decisionTreeDS.csv"
dataset = pd.read_csv(filename) # Check file extension before using this function
X = dataset[['OverallQual', 'GrLivArea', 'GarageCars']].values

y = dataset['SalePrice'].values

###################### 2- Preprocessing ######################
# preprocess

# Split data
test_train_ratio = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_train_ratio)


###################### 3- Training ######################
class decision_tree_regressor:

  def fit(self, X, y, min_leaf = 5):

    self.dtree = Node(X, y, np.array(np.arange(len(y))), min_leaf)

    return self

  def predict(self, X):

    return self.dtree.predict(X)


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

class Node:

    def __init__(self, x, y, idxs, min_leaf=5):

        self.x = x

        self.y = y

        self.idxs = idxs

        self.min_leaf = min_leaf

        self.row_count = len(idxs)

        self.col_count = x.shape[1]

        self.val = np.mean(y[idxs])

        self.score = float('inf')

        self.find_varsplit()

    def find_varsplit(self):

        for c in range(self.col_count): self.find_better_split(c)

        if self.is_leaf: return

        x = self.split_col

        lhs = np.nonzero(x <= self.split)[0]

        rhs = np.nonzero(x > self.split)[0]

        self.lhs = Node(self.x, self.y, self.idxs[lhs], self.min_leaf)

        self.rhs = Node(self.x, self.y, self.idxs[rhs], self.min_leaf)

    def find_better_split(self, var_idx):

        x = self.x[self.idxs, var_idx]

        for r in range(self.row_count):

            lhs = x <= x[r]

            rhs = x > x[r]

            if rhs.sum() < self.min_leaf or lhs.sum() < self.min_leaf: continue

            curr_score = self.find_score(lhs, rhs)

            if curr_score < self.score:

                self.var_idx = var_idx

                self.score = curr_score

                self.split = x[r]

    def find_score(self, lhs, rhs):

        y = self.y[self.idxs]

        lhs_std = y[lhs].std()

        rhs_std = y[rhs].std()

        return lhs_std * lhs.sum() + rhs_std * rhs.sum()

    @property

    def split_col(self): return self.x[self.idxs,self.var_idx]

    @property

    def is_leaf(self): return self.score == float('inf')

    def predict(self, x):

        return np.array([self.predict_row(xi) for xi in x])

    def predict_row(self, xi):

        if self.is_leaf: return self.val

        node = self.lhs if xi[self.var_idx] <= self.split else self.rhs

        return node.predict_row(xi)
    
model = decision_tree_regressor().fit(X_train, y_train)

preds = model.predict(X_test)

###################### 4- Testing ######################
model_score = model.score(X_test, y_test)


###################### 5- Visualization ######################
plt.scatter(X_test, y_test, color="red")
plt.plot(X_test, model.predict(X_test), color="blue")
plt.show()


###################### 6- Save & Use ######################
values_to_predict = X_test
prediction_result = model.predict([ values_to_predict ])

with open('linearModel.pkl', 'wb') as f:
    pickle.dump(model, f)
