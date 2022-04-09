import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

data_source = 'http://bit.ly/w-data'
data = pd.read_csv(data_source)
data.head(10)

X = data['Hours'].values
Y = data['Scores'].values

plt.scatter(X, Y)
plt.title('Hours vs Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Score')
plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

X = np.expand_dims(X, axis=1)
Y = np.expand_dims(Y, axis=1)
X_train = np.expand_dims(X_train, axis=1)
Y_train = np.expand_dims(Y_train, axis=1)
X_test = np.expand_dims(X_test, axis=1)
Y_test = np.expand_dims(Y_test, axis=1)

model = LinearRegression()
model.fit(X_train, Y_train)
line = model.coef_ * X + model.intercept_
prediction = model.predict(X_test)

plt.scatter(X, Y)
plt.plot(X, line)
plt.show()

hours = float(input('Enter Number of Hours: '))
test = np.array([hours])
test = np.expand_dims(test, axis=1)
test = model.predict(test)
print("Number of Hours: " + str(hours))
print("Predicted Score: " + str(test[0][0]))

print('Mean Absolute Error: ', metrics.mean_absolute_error(Y_test, prediction))
