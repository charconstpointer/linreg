from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import math
X = []
y = []

for _ in range(100000):
    X.append([_])
    y.append([math.log10(_+1)])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
reg = LinearRegression().fit(X_train, y_train)
print(reg.score(X_train, y_train))
plt.scatter(X_test, y_test, color='green')
plt.scatter(X, y, color='red')
plt.plot(X_train, reg.predict(X_train), color='blue')
print(reg.predict([[4]]))
plt.show()
