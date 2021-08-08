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
filename = "Mall_Customers.csv"
dataset = pd.read_csv(filename) # Check file extension before using this function
X = dataset.iloc[:, 2:].values


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
#model = KMeans(n_clusters = K, init = 'k-means++')
#model.fit(X)


class K_Means:
	def __init__(self, k =3, tolerance = 0.0001, max_iterations = 500):
		self.k = k
		self.tolerance = tolerance
		self.max_iterations = max_iterations

	def fit(self, data):

		self.centroids = {}

		#initialize the centroids, the first 'k' elements in the dataset will be our initial centroids
		for i in range(self.k):
			self.centroids[i] = data[i]

		#begin iterations
		for i in range(self.max_iterations):
			self.classes = {}
			for i in range(self.k):
				self.classes[i] = []

			#find the distance between the point and cluster; choose the nearest centroid
			for features in data:
				distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]
				classification = distances.index(min(distances))
				self.classes[classification].append(features)

			previous = dict(self.centroids)

			#average the cluster datapoints to re-calculate the centroids
			for classification in self.classes:
				self.centroids[classification] = np.average(self.classes[classification], axis = 0)

			isOptimal = True

			for centroid in self.centroids:

				original_centroid = previous[centroid]
				curr = self.centroids[centroid]

				if np.sum((curr - original_centroid)/original_centroid * 100.0) > self.tolerance:
					isOptimal = False

			#break out of the main loop if the results are optimal, ie. the centroids don't change their positions much(more than our tolerance)
			if isOptimal:
				break

	def pred(self, X):
		preds = []
		for data in X:
			distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
			classification = distances.index(min(distances))
			preds.append(classification)
		return np.array(preds)

model = K_Means(8)
model.fit(X)
###################### 4- Testing ######################
y = model.pred(X)


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