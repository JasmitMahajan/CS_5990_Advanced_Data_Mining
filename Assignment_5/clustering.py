#-------------------------------------------------------------------------
# AUTHOR: Jasmit Mahajan
# FILENAME: clustering.py
# FOR: CS 5990- Assignment #5
# TIME SPENT: 1 hr 30 min
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

df = pd.read_csv('training_data.csv', sep=',', header=None) #reading the data by using Pandas library

#assign your training data to X_training feature matrix
X_training = np.array(df.values)
#run kmeans testing different k values from 2 until 20 clusters
     #Use:  kmeans = KMeans(n_clusters=k, random_state=0)
     #      kmeans.fit(X_training)
     #--> add your Python code
first = []
second = []

max_score = 0
max_k = 0

#run kmeans testing different k values from 2 until 20 clusters
for k in range(2, 21):
    k_means = KMeans(n_clusters=k, random_state=0)
    k_means.fit(X_training)

     #for each k, calculate the silhouette_coefficient by using:
    score = silhouette_score(X_training, k_means.labels_)

    if score > max_score:
        max_k = k

    max_score = max(max_score, score)

    first.append(k)
    second.append(score)

#plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
plt.plot(first, second)

#reading the validation data (clusters) by using Pandas library
#--> add your Python code here
testing_data = pd.read_csv('testing_data.csv', header=None)

#assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
# do this: np.array(df.values).reshape(1,<number of samples>)[0]
#--> add your Python code here
labels = np.array(testing_data.values).reshape(1, len(testing_data.values))[0]

#Calculate and print the Homogeneity of this kmeans clustering
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, k_means.labels_).__str__())
#--> add your Python code here
