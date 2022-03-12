from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
model.score
metrics.accuracy_score(y_test, model.predict(X_test))

i = 1
x = X_test[i]
x_new = np.array([x])
print("\n XNEW \n", x_new)

for i in range(len(X_test)):
    x = X_test[i]
    x_new = np.array([x])
    prediction = model.predict(x_new)
    print("\n Actual : {0} {1}, Predicted :{2}{3}".format(y_test[i], iris["target_names"][y_test[i]], prediction,
                                                          iris["target_names"][prediction]))

print("\n TEST SCORE[ACCURACY]: {:.2f}\n".format(model.score(X_test, y_test))) 
