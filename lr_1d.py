import numpy as np
import matplotlib.pyplot as plt

#load data
X = []
Y = []
for line in open('data_1d.csv'):
	x, y = line.split(',')
	X.append(float(x))
	Y.append(float(y))

#convert X and Y into numpy arrays
X = np.array(X)
Y = np.array(Y)

#plot data
plt.scatter(X,Y)
plt.show()

#apply loss function
denominator = X.dot(X) - X.mean() * X.sum()
a = (X.dot(Y) - Y.mean() * X.sum()) / denominator
b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y) )  / denominator

#compute y-hat (predicted y)
Yhat = a*X + b

#plot predicted line
plt.scatter(X,Y)
plt.plot(X, Yhat)
plt.show()

#compute r-squared
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)

print("R-SQUARED is:", str(r2))