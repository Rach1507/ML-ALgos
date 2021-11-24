import numpy as np
import pandas as pd

df = pd.read_csv("weatherid3.csv")
df.head()

from sklearn.model_selection import train_test_split

X = df[['Outlook', 'Temperature']]
y = df['PlayTennis']
from sklearn import preprocessing

print(X)

from sklearn.preprocessing import LabelEncoder

X1 = X.apply(LabelEncoder().fit_transform)
print(X1)
le = preprocessing.LabelEncoder()
y1 = le.fit_transform(y)
print(y1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.33, random_state=42)
print('Training Data Shape:', X_train.shape)
print('Testing Data Shape:', X_test.shape)

from sklearn.naive_bayes import GaussianNB

naive_model = GaussianNB()
naive_model.fit(X_train, y_train)
from sklearn import metrics

predictions = naive_model.predict(X_test)
print(metrics.classification_report(y_test, predictions))
df = pd.DataFrame(metrics.confusion_matrix(y_test, predictions), index=['yes', 'no'], columns=['yes ', 'no'])
print(df)
print(metrics.accuracy_score(y_test, predictions)) 
