# Import pre-processing libs
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hc
from scipy.spatial import distance
import math
#from sklearn.preprocessing import OneHotEncoder, LabelEncoder
#from sklearn.preprocessing import StandardScaler


# Import model
from sklearn.cluster import AgglomerativeClustering

# Import post-processing libs
import matplotlib.pyplot as plt
import pickle


###################### 1- Import Data ######################
filename = "Mall_Customers.csv"
dataset = pd.read_csv(filename) # Check file extension before using this function
X = dataset.iloc[:, 2:]


###################### 2- Preprocessing ######################

# Find out the best values of K using Dendograms 
# & applying Hierarchial clustering to a set of numbers for k
dendrogram = hc.dendrogram(hc.linkage(X, method = 'ward', metric = 'euclidean'))
plt.title('Dendrogram')
plt.xlabel('')
plt.ylabel('Distances')
plt.show()

###################### 3- Training ######################
K = 2
model = AgglomerativeClustering(n_clusters = K, affinity = 'euclidean', linkage = 'ward')


###################### 4- Testing ######################
y = model.fit_predict(X)


###################### 5- Visualization ######################
###### IMPORTANT NOTE: this visualization works for 2 dimensions only ######
colors = ['red', 'blue', 'lightcoral', 'indigo', 'gold', 'crimson', 'fuchsia', 'peru', 'palegreen', 'lawngreen', 'olivedrab', 'yellow', 'darkseagreen', 'tomato', 'orange', 'darkgreen', 'springgreen', 'darkred', 'teal', 'midnightblue', 'brown', 'gray', 'darkviolet', 'aqua', 'purple', 'orangered', 'turquoise', 'dodgerblue', 'deeppink']
for i in range(K):
    plt.scatter(X[y == i, 0], X[y == i, 1], s = 100, c = colors[i], label = 'Cluster ' + str(i + 1))
plt.title('')
plt.xlabel('')
plt.ylabel('')
plt.legend()
plt.show()


###################### 6- Save & Use ######################
values_to_predict = X
prediction_result = model.predict(values_to_predict)

with open('H_clustering.pkl', 'wb') as f:
    pickle.dump(model, f)