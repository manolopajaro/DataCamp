

import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

newsgroups = sklearn.datasets.fetch_20newsgroups_vectorized()
X, y = newsgroups.data, newsgroups.target
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
knn = KNeighborsClassifier(n_neighbors=1) #we set hyperparameter: number of neighbors = 1
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
score = knn.score(X_test,y_test)
print(score)