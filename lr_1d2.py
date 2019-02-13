import numpy as np
import matplotlib.pyplot as plt
import time

#load data
X = []
Y = []
for line in open('data_1d.csv'):
	x, y = line.split(',')
	X.append(float(x))
	Y.append(float(y))

#convert X and Y into numpy arrays
X = np.random.random_sample((1000000000,))
Y = np.random.random_sample((1000000000,))

#apply loss function
start2 = time.time()
denominator2 = X.dot(X) - X.mean() * X.sum()
a2 = (X.dot(Y) - Y.mean() * X.sum()) / denominator2
b2 = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y) )  / denominator2
end2 = time.time()
print(end2 - start2)