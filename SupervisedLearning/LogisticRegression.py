import sklearn.datasets
from sklearn.linear_model import LogisticRegression

wine = sklearn.datasets.load_wine()
lr=LogisticRegression()
lr.fit(wine.data, wine.target)
print(lr.score(wine.data, wine.target))
wine_data1=wine.data[:1]
print(lr.predict_proba(wine_data1))
