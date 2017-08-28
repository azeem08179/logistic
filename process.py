import numpy as np 
import pandas as pd 
from colorama import Fore, Back, Style  #pip install colorama - to change the text color 

def get_data():

    df = pd.read_csv('ecommerce_data.csv')
    print(Fore.RED + '###### Loading Data ...... ##############\n')
    print(Style.RESET_ALL)
    print df.head()
    data = df.as_matrix()

    X = data[:, :-1] # copying all elements except last column
    Y = data[:,-1]   # all rows of last column

    # Normalizing columns with headers   "n_products_viewed" , " visit_duration "

    X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std() # n_products_viewed cloumn
    X[:,2] = (X[:,2] - X[:,2].mean()) / X[:,2].std() # visit_duration column

    N, D = X.shape
    print "\nNumber of Samples in data = ",N
    print "\nNumber of Features in data = ",D

    X2 = np.zeros((N,D+3))
    N2, D2 = X2.shape
    print "\nNumber of Samples in X2 = ",N2
    print "\nNumber of Features in X2 = ",D2

    X2[:,0:(D-1)] = X[:,0:(D-1)]  # non-categorical  D=5

    print "\nX2 = ", np.array_str(X2[1,:], precision=2)
    print("X2 (time of the day coloumn):"),X[1:50,4]
    # one hot encoding for time of the day variable
    # D-1 = 4  4th coloumn is the time of the day
    # columns will be 4, 5 , 6 , 7
    for n in xrange(N):
       t = int(X[n,D-1])
       X2[n,t+D-1] = 1
        
        #Z = np.zeros((N,4))
        #Z[np.arrange(N),X[:,D-1].astype(np.int32)] = 1

        #assert(np.abs(X2[:,-4:])- Z).sum() < 10e-10 )
    print(Fore.RED + '***** X2 sample after one hot encoding:****\n')
    print(Style.RESET_ALL)
    print X2[1:20,4:8] # debug  
    return X2, Y



def get_binary_data():
    X, Y = get_data()
    X2 = X[Y <= 1]
    Y2 = Y[Y <= 1]  
    return X2, Y2

