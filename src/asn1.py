import numpy as np
import csv
import random
import itertools
    

    # ##############################################################################
    # Dataset preparation
    # ##############################################################################

def read_data(path, selector = lambda row: [eval(x) for x in row], skipFirstRow = False): 
    """
    Read csv dataset from file.
    :param path: path name to the dataset file
    :param selector: func applied to each row. Primarily for migraine data set to filter out comments and dates.
    :returns: A single array for the dataset
    """           
    reader = csv.reader(open(path, 'r'))
    data = []
	
    if skipFirstRow:	
        reader.next()  # skip the first one
    skips = 0
    for row in reader:
        try: 
           data.append(selector(row))
        except:
            #print selector(row)
            skips += 1
            pass # Throw away recognized. There are some empty rows.
            
    print skips, "rows skipped"
    return np.array(data)

def seperate_data(data, y = 0):
    """
    Process the dataset after reading it. Seperate it into components and calculate some derived quantities.
    :param data: Data set read in from read_data
    :param y: index of output feature.
    :returns: None
    """
    Y = data[:,y]
    X_NoBias = np.hstack((data[:,:y],data[:,y+1:]))
    Bias = np.ones((X_NoBias.shape[0],1))
    X = np.hstack((Bias, X_NoBias))
    return X, Y     

def training_testing(X,Y, m = None):
    """
    Split the dataset into two. (i.e. training and testing)
    :param X: input vectors
    :param Y: output values
    :param m: Desired size of training set. Defaults to a 50/50 split.
    """

    if m == None:
        m = Y.shape[0]/2
            
    nums = range(Y.shape[0])
    random.shuffle(nums)
    
    X_train = np.array([X[i] for i in nums[:m]])
    X_test  = np.array([X[i] for i in nums[m:]])
    Y_train = np.array([Y[i] for i in nums[:m]])
    Y_test  = np.array([Y[i] for i in nums[m:]])     
    return X_train, Y_train, X_test, Y_test

def training_testing_equalized(X,Y, m = None):
    sev = [ i for i in range(len(X)) if Y[i] == 2 ]
    mld = [ i for i in range(len(X)) if Y[i] == 1 ]
    nom = [ i for i in range(len(X)) if Y[i] == 0 ]
    
    random.shuffle(sev)
    random.shuffle(mld)
    random.shuffle(nom)
    
    if m == None or m > len(sev) or m > len(mld) or m > len:
        m = min(len(sev), len(mld), len(nom)) /2 
    
    train = sev[:m] + mld[:m] + nom[:m]
    test  = sev[m:] + mld[m:] + nom[m:]
    
    random.shuffle(train)
    random.shuffle(test)
    
    X_train = np.array([X[i] for i in train])
    X_test  = np.array([X[i] for i in test])
    Y_train = np.array([Y[i] for i in train])
    Y_test  = np.array([Y[i] for i in test])     
    return X_train, Y_train, X_test, Y_test    
        

def square_terms(X):
    """
    Add all square terms to each vector.
    :param X: An input dataset (with bias)
    :returns: New dataset with each possible multiplication of terms.
    """
    newterms = []
    for vector in X:
        newterms.append( [x * y for x,y in itertools.combinations_with_replacement(vector, 2)] )
    return np.array(newterms)

    # ##############################################################################
    # Regression
    # ##############################################################################

    # Theta vector generators

def theta_0(n):
    """
    :param n: Size of feature vector
    :returns: An initial theta guess of all zeroes for gradient descent
    """
    return np.zeros((n, ))

def theta_1(n):
    """
    :param n: Size of feature vector
    :returns: An initial theta guess of all ones for gradient descent
    """
    return np.ones((n, ))
    
def theta_r(n, r = 1.):
    """
    :param n: Size of feature vector
    :param r: Maximum value for r.
    :returns: An initial theta guess of random values for gradient descent
    """
    return np.array( [ r * random.random() for i in range(n) ] )                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      

    # Per vector operations (hypothesis and cost)

def g(z):
    """
    The logistic function.
    :param z: real number (-inf, +inf)
    :returns: real number (-1, 1)
    """
    return 1. / (1 + np.e ** z)   

def H_log(theta, x):
    """
    Logistic hypothesis function.
    :param theta: weight vector
    :param x: one input vector
    :returns: predicted class of vector x
    """
    return g(H_lin(theta, x))

def H_lin(theta, x):
    """
    Linear hypothesis function.
    :param theta: weight vector
    :param x: one input vector
    :returns: predicted output value for vector x
    """
    return theta.T.dot(x)

def Cost_lin(hx, y):
    """
    Linear cost of prediction vs. reality.
    :param hx: predicted value for a vector x. (i.e. from H_lin)
    :param y: known output value for that vector
    :returns: Cost of the prediction. How far off was the hypothesis?
    """
    return (hx - y)**2  

def Cost_log(hx, y):
    """
    Logistic cost of prediction vs. reality
    :param hx: predicted class for a vector x. (i.e. from H_log)
    :param y: known output class for that vector
    :returns: Cost of the prediction. How far off was the hypothesis?
    """
    return -y * np.log(hx) - (1-y) * np.log(1 - hx)
     
    # Total cost for dataset (J) and partial derivatives
          
def Jay(X, Y, theta, cost = None, h = None, reg = 0):
    """
    Total cost for the entire dataset of a prediction theta.
    :param X: input vectors
    :param Y: output vectors
    :param theta: learnt vector
    :param cost: cost function. Either Cost_lin or Cost_log
    :param h: H function. Either H_lin or H_log
    :param reg: Regularization factor
    """
    if cost == None:
        cost = Cost_lin
    if h == None:
        h = H_lin
    m = Y.shape[0]  
    return 1./(2*m) * ( sum( [ cost(h(theta, X[i]), Y[i]) for i in range(m) ] ) ) + reg * sum(x**2 for x in theta[1:]) 
    
def ddJ_lin(X, Y, theta, j):
    """
    Linear partial derivatives w.r.t j
    :param X: input vectors
    :param Y: output value
    :param theta: learnt vector
    :param j: index of feature
    :returns: 
    """
    m = Y.shape[0]
    return 1./m * sum([ (H_lin(theta, X[i]) - Y[i]) * X[i][j] for i in range(m) ]) 
    
def ddJ_log(X, Y, theta, j):
    """
    Logistic partial derivatives w.r.t j
    :param X: input vectors
    :param Y: output value
    :param theta: learnt vector
    :param j: index of feature
    :returns: 
    """
    m = Y.shape[0]
    return 1./m * sum([ ((H_log(theta, X[i]) - Y[i]) * X[i][j]) for i in range(m) ])


    # ##############################################################################
    # Theta optimizations
    # ##############################################################################
            
    # Gradient Descent
            
def grad_descent_step(X, Y, theta, rate, d = ddJ_lin,  reg = 0):
    """
    One iteration of gradient descent
    :param X: set of input vectors
    :param Y: set of output values
    :param theta: weight vector
    :param rate: Learning rate
    :param d: partial derivative of J
    :param reg: Regularization constant. Defaults to 0 (no regularization)
    :returns: new weight vector
    """
    theta2 = theta[:]
    m = Y.shape[0]
    for i in range(len(theta)):
        if i == 0: # Don't regularize the bias.
            theta2[i] = theta[i] - rate * d(X,Y, theta, i)
        else:
            theta2[i] = theta[i] * ( 1 - rate * reg / m)  - rate * d(X,Y, theta, i)
    return np.array( theta2 )
    
def grad_descent(X, Y, iterations, rate, theta = None, linear = True, reg = 0):
    """
    Fixed number of gradient descent steps
    :param X: set of input vectors
    :param Y: set of output values
    :param iterations: number of gradient descent steps to take
    :param rate: Learning rate
    :param theta: starting weight vector. Defaults to a randomly generated theta.
    :param linear: boolean whether to use linear or logistic cost/hypothesis function. Defaults to True (linear).
    :param reg: Regularization coefficient. Defaults to 0 (no regularization).
    :returns: new weight vector
    """
    if theta == None:
        theta = theta_r(X.shape[1])
    d = ddJ_lin if linear else ddJ_log
    
    for i in range(iterations):    
        theta = grad_descent_step(X, Y, theta, rate, d, reg)
        
    return theta

    # Normal Equation

def normal_eqn(X, Y, reg = 0):
    """
    Numerical solution to theta.
    :param X: set of input vectors
    :param Y: set of output values
    :param reg: Regularization coefficient. Defaults to 0 (no regularization).
    :returns: weight vector
    """    
    a = reg * np.eye(X.shape[1])
    a[0][0] = 0 # Don't regularize the bias.    
    return np.linalg.pinv( X.T.dot( X ) + a ).dot( X.T ).dot( Y )
    
