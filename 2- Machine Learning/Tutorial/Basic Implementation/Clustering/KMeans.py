# Import pre-processing libs
import numpy as np
import pandas as pd
#from sklearn.preprocessing import OneHotEncoder, LabelEncoder
#from sklearn.preprocessing import StandardScaler

# Import model
from sklearn.cluster import KMeans

# Import post-processing libs
import matplotlib.pyplot as plt
import pickle


###################### 1- Import Data ######################
filename = ""
dataset = pd.read_csv(filename) # Check file extension before using this function
X = dataset.iloc[:, :-1].values


###################### 2- Preprocessing ######################

# Find out the best values of K using WCSS formula 
# & applying KMeans to a set of numbers for k
wcss = []
minK, maxK = (1, 10)
Krange = range(minK, maxK + 1)

for k in Krange:
    kmeans = KMeans(n_clusters = k, init = 'k-means++',)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(Krange, wcss)
plt.title('Elbow Method')
plt.xlabel('N of clusters')
plt.ylabel('WCSS')
plt.show()


###################### 3- Training ######################
K = 8
model = KMeans(n_clusters = K, init = 'k-means++')
model.fit(X)


###################### 4- Testing ######################
y = model.predict(X)


###################### 5- Visualization ######################
###### IMPORTANT NOTE: this visualization works for 2 dimensions only ######
colors = ['red', 'blue', 'lightcoral', 'indigo', 'gold', 'crimson', 'fuchsia', 'peru', 'palegreen', 'lawngreen', 'olivedrab', 'yellow', 'darkseagreen', 'tomato', 'orange', 'darkgreen', 'springgreen', 'darkred', 'teal', 'midnightblue', 'brown', 'gray', 'darkviolet', 'aqua', 'purple', 'orangered', 'turquoise', 'dodgerblue', 'deeppink']
for i in range(K):
    plt.scatter(X[y == i, 0], X[y == i, 1], s = 100, c = colors[i], label = 'Cluster ' + str(i + 1))
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s = 200, c ='black', label = 'Centroids')
plt.title('')
plt.xlabel('')
plt.ylabel('')
plt.legend()
plt.show()


###################### 6- Save & Use ######################
values_to_predict = X
prediction_result = model.predict(values_to_predict)

with open('kmeans.pkl', 'wb') as f:
    pickle.dump(model, f)