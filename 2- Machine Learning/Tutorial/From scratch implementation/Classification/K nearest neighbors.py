# Import pre-processing libs
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from operator import itemgetter

# Import model
from sklearn.neighbors import KNeighborsClassifier

# Import post-processing libs
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
import pickle


###################### 1- Import Data ######################
filename = 'Social_Network_Ads.csv'
dataset = pd.read_csv(filename)
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
model = KNeighborsClassifier(n_neighbors = 5, p = 2, metric = 'minkowski')
model.fit(X_train, y_train)


class Point:

    def __init__(self, axis):
        """Point constructor
        :param axis: iterable with point coordinates
            :type: list, tuple and np.array
            :example:
                        x    y    z     w
                axis = [1, 0.3, 6.4, -0.2]
        """

        self.axis = np.array(axis)

    def distance(self, other):
        """Euclidean distance between 2 points with n dimensions
        :param other: coordinates with same dimension of self
            :type: Point, list, tuple, np.array
            :example:
                np.array([-7, -4, 3])
                   Point(-7, -4, 3)
                        [-7, -4, 3]
                        (-7, -4, 3)
        :return: euclidean distance
            :type: float
        """

        if not isinstance(other, Point):
            other = Point(other)
        # Euclidean distance
        return sum((self - other) ** 2) ** 0.5

    def to_numpy(self):
        """Point class to np.array
        :return: self.axis
            :type: np.array
        """
        return self.axis

    def to_list(self):
        """Point class to list
        :return: self.axis.tolist()
            :type: list
        """
        return self.axis.tolist()

    def __add__(self, other):
        if isinstance(other, Point):
            return Point(self.axis + other.axis)
        return Point(self.axis + np.array(other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Point):
            return Point(self.axis - other.axis)
        return Point(self.axis - np.array(other))

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        if isinstance(other, Point):
            return Point(self.axis * other.axis)
        return Point(self.axis * np.array(other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Point):
            return Point(self.axis / other.axis)
        return Point(self.axis / np.array(other))

    def __rtruediv__(self, other):
        return self.__truediv__(other)

    def __floordiv__(self, other):
        if isinstance(other, Point):
            return Point(self.axis // other.axis)
        return Point(self.axis // np.array(other))

    def __rfloordiv__(self, other):
        return self.__floordiv__(other)

    def __pow__(self, power, modulo=None):
        if modulo:
            return self.axis ** power % modulo
        return self.axis ** power

    def __eq__(self, other):
        if isinstance(other, Point):
            return max(self.axis == other.axis)
        return max(self.axis == other)

    def __getitem__(self, item):
        return self.axis[item]

    def __repr__(self):
        return f'Point{tuple(self.axis)}'

class KNearestNeighbors:

    def __init__(self, k=3):
        """KNearestNeighbors constructor
        :param k: total of neighbors
            :type: int
        """
        self.k = int(k)
        self._fit_data = []

    def fit(self, x, y):
        """Train knn model with x data
        :param x: data with coordinates
            :type: list, tuple, np.array
            :example:
                [[1, 2], [2, 3], [3, 4], [4, 5]]
        :param y: labels to x data set
            :type: list, tuple, np.array
            :example:
                [[1], [0], [0], [1]]
        :return: None
        """

        assert len(x) == len(y)
        # [(Point(1, 2), [0]), (Point(0, -1), [0]), (Point(5, 5), [1])]
        self._fit_data = [(Point(coordinates), label) for coordinates, label in zip(x, y)]

    def predict(self, x):
        """Predict x array
        :param x: data with coordinates to be predicted
            :type: list, tuple, np.array
            :example:
                [[1, 2], [2, 3], [3, 4], [4, 5]]
        :return: x predicts
            :type: list
            :example:
                [[1], [0], [0], [1]]
        """

        predicts = []
        for coordinates in x:
            predict_point = Point(coordinates)

            # euclidean distance from predict_point to all in self._fit_data
            distances = []
            for data_point, data_label in self._fit_data:
                distances.append((predict_point.distance(data_point), data_label))

            # k points with less distances
            distances = sorted(distances, key=itemgetter(0))[:self.k]
            # label of k points with less distances
            predicts.append(list(max(distances, key=itemgetter(1)))[1])

        return np.array(predicts)
    
    def evaluate(self, y_true, y_pred):
        if y_true.shape[0] == y_pred.shape[0]:
            return np.count_nonzero( (y_true == y_pred) ) / y_pred.shape[0]
        else:
            print("y_true and y_pred must have the same dimensions")


model = KNearestNeighbors(k=5)
model.fit(X_train, y_train)



###################### 4- Testing ######################

#model_score = model.score(X_test, y_test)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
score = model.evaluate(y_test, y_pred)

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