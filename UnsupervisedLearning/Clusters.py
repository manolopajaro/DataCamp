import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Import KMeans
from sklearn.cluster import KMeans
import test_data

""""
xs = new_points[:,0]
ys = new_points[:,1]
plt.scatter(xs,ys)
xs = points[:,0]
ys = points[:,1]
plt.scatter(xs,ys)
plt.show()
"""

if __name__ == "__main__":
    test_data.initialize()
    # Create a KMeans instance with 3 clusters: model
    model = KMeans(n_clusters=3)

    # Fit model to points
    model.fit(test_data.points)

    # Determine the cluster labels of new_points: labels
    labels = model.predict(test_data.new_points)

    # Print cluster labels of new_points
    #print(labels)

    xs = test_data.new_points[:,0]
    ys = test_data.new_points[:,1]

    # Make a scatter plot of xs and ys, using labels to define the colors
    plt.scatter(xs,ys, c=labels, alpha=0.5)

    # Assign the cluster centers: centroids
    centroids = model.cluster_centers_

    # Assign the columns of centroids: centroids_x, centroids_y
    centroids_x = centroids[:,0]
    centroids_y = centroids[:,1]

    # Make a scatter plot of centroids_x and centroids_y
    plt.scatter(centroids_x,centroids_y,marker='D', s=50)
    plt.show()
