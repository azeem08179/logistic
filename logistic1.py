import numpy as np


N = 100
D = 2

X = np.random.randn(N,D)
#print(X)

ones = np.array([[1]*N]).T
Xb = np.concatenate((ones,X), axis = 1)

w = np.random.randn(D+1)

print 'Shape of Xb:',Xb.shape
print 'Shape of w:',w.shape

z = Xb.dot(w)

print('Before sigmoid z='),z

def sigmoid(z):
	return 1/(1 + np.exp(-z))


print('After sigmoid z=')
print sigmoid(z)