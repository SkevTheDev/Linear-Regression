import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# load data
X = []
Y = []
for line in open('data_2d.csv'):
	x1, x2, y = line.split(',')
	X.append([1, float(x1), float(x2)])
	Y.append(float(y))

# convert X and Y into numpy arrays
X = np.array(X)
Y = np.array(Y)
X1 = np.random.random_sample([100,])

print(X.shape)

print(X1)

X = np.c_[X, X1]

print(X.shape)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X[:,0], X[:,1], Y)
# plt.show()

# compute weights
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat = np.dot(X, w)

# compute R-SQUARED
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)

print("R-SQUARED is:", str(r2))