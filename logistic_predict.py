import numpy as np 
from process import get_binary_data

X, Y = get_binary_data() # data after one hot encoding

D = X.shape[1] # D is the no of features

w = np.random.randn(D) # w is a cloumn vector

b = 0

def sigmoid(a):
	return 1 / (1 + np.exp(-a))



def forward(X, w, b):
	return sigmoid(X.dot(w) + b)

P_Y_given_X = forward(X, w, b)
print("P_Y_given_X=\n"),P_Y_given_X[1:10]	
predictions = np.round(P_Y_given_X)
print("predictions=\n"),predictions[1:10]

def classification_rate(Y,P):
	return np.mean(Y == P)

print "Score",classification_rate(Y,predictions)	
