from sklearn.datasets import make_moons
from sklearn.cluster import KMeans

import pandas as pd

test = pd.read_csv("source/test_moon.csv")

x = test[["X1", "X2"]]

X, y = make_moons(n_samples=1000, noise=.05, random_state=1004)
kmeans = KMeans(n_clusters=2)

kmeans.fit(X, y)

print(kmeans.predict(x))
