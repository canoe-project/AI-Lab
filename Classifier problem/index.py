from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

breast = load_breast_cancer()

x = breast.data
y = breast.target

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1004)

knn = KNeighborsClassifier(n_neighbors=4)

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
scores = metrics.accuracy_score(y_test, y_pred)

print(scores)
