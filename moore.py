import re
import numpy as np
import matplotlib.pyplot as plt

X = []
Y = []

non_decimal = re.compile(r'[^\d]+')

for line in open('moore.csv'):
	r = line.split('\t')
	x = int(non_decimal.sub('', r[2].split('[')[0]))
	y = int(non_decimal.sub('', r[1].split('[')[0]))
	X.append(x)
	Y.append(y)

X = np.array(X)
Y = np.array(Y)

plt.scatter(X, Y)
plt.show()

Y = np.log(Y)
plt.scatter(X,Y)
plt.show()

#apply loss function
#denominator = X.dot(X) - X.mean() * X.sum()
#a = (X.dot(Y) - Y.mean() * X.sum()) / denominator
#b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y) )  / denominator

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
print("a:",  a, "b:", b)
print("R-SQUARED is:", str(r2))

#derivation to compute how long it takes for transistor count to double
#	log(tc) = a*year +b
#	tc = exp(b) * exp(a * year)
#	2*tc = 2 * exp(b) * 2 * exp(a * year) = exp(ln(2)) * exp(b) * exp(a * year)
#			 = exp(b) * exp(a * year + ln(2))
# exp(b) * exp(a*year2) = exp(a*year1 + ln2)
# a*year2 = a*year1 + ln2
# year2 = year1 + ln2/a
print("time to double: ", np.log(2)/a)