import numpy as np
import pandas as pd
#from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Import model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression


# Import post-processing libs
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix

###################### 1- Import Data ######################

filename = "Alcohol.csv"
dataset = pd.read_csv(filename) # Check file extension before using this function
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
#y = y.reshape(y.shape[0], 1)

n_classes = np.unique(y).shape[0]

###################### 2- Preprocessing ######################

# Split data
test_train_ratio = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_train_ratio)

# SCALING
sc_x = StandardScaler()
sc_y = StandardScaler()

# Scale X
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

# Scale y



###################### 3- Training ######################
# PCA
n_components = 2
lda = LDA(n_components = n_components)
_X_train = lda.fit_transform(X_train, y_train)
_X_test = lda.fit_transform(X_test, y_test)


# Logistic Regression
classifier = LogisticRegression()
classifier.fit(_X_train, y_train)


###################### 3- Testing ######################
y_pred = classifier.predict(_X_test)
cm = confusion_matrix(y_test, y_pred)


###################### 3- Visualization ######################

colors = ['red', 'blue', 'lightcoral', 'indigo', 'gold', 'crimson', 'fuchsia', 'peru', 'palegreen', 'lawngreen', 'olivedrab', 'yellow', 'darkseagreen', 'tomato', 'orange', 'darkgreen', 'springgreen', 'darkred', 'teal', 'midnightblue', 'brown', 'gray', 'darkviolet', 'aqua', 'purple', 'orangered', 'turquoise', 'dodgerblue', 'deeppink']
selected_colors = colors[0 : n_classes]

X_set, y_set = _X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(selected_colors))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(selected_colors)(i), label = j)

plt.title('Logistic Regression (Training set)')
plt.xlabel('')
plt.ylabel('')
plt.legend()
plt.show()


X_set, y_set = _X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(selected_colors))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(selected_colors)(i), label = j)

plt.title('Logistic Regression (Test set)')
plt.xlabel('')
plt.ylabel('')
plt.legend()
plt.show()



