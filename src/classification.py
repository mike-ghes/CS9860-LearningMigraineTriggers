from asn1 import *
import matplotlib.pyplot as plt
import numpy as np

def load_wine():
    """
    Load wine dataset.
    """
    data = read_data('wine.csv')
    return seperate_data(data)


def multiclass_test(X2, Y2, thetas):
    
    debug = False # Setting flag to true will cause mis-classifications to be printed. 
    results = []
    for i in range(len(X2)):
        h = [H_log(theta, X2[i]) for theta in thetas]
        hypoth = {  0:1.0,
                    1:2.0,
                    2:3.0  }[h.index(min(h))] 
        if debug and Y2[i] != hypoth:
            print "actual: ", Y2[i], "hypothesis: ", hypoth, "-----",  h
        results += [Y2[i] == hypoth]                       
    
    return results

def multi_normal(X, Ym, reg = 0):
    out = []
    for partialY in Ym:
        out.append( normal_eqn(X, partialY, reg) )
    return out

X,Y = load_wine()

def training_size_vs_classification(reg = 0, col = 'b', sqterms = False):
    out = {}
    for i in range(1600):
    
        size = i / 10 + 10
        X_train,Y_train,X_test,Y_test = training_testing(square_terms(X) if sqterms else X,Y,size)
        
        Ym = [ np.array([int(y == 1) for y in Y_train]), np.array([int(y == 2) for y in Y_train]), np.array([int(y == 3) for y in Y_train]) ]
        thetas = multi_normal(X_train, Ym, reg)    
        results = multiclass_test(X_test, Y_test, thetas)
        
        x = 1 - float(sum(results))/len(results)
        y = size
        
        if (x,y) in out:
            out[(x,y)] += 1
        else:
            out[(x,y)] = 1
    
    x2 = [x[0] for x in out.keys()]
    y2 = [x[1] for x in out.keys()]
    sz = [x**2 for x in out.values()]
    
    plt.scatter(y2, x2, s = sz, c = col)
    plt.title('Training set size vs. misclassification (reg = ' + str(reg) + ')')
    plt.xlabel('Training set size')
    plt.ylabel('Percent Misclassification')
    plt.ylim(0,1)
    plt.show()
    
def figures():
    plt.figure(1) 
    training_size_vs_classification()
    plt.figure(2) 
    training_size_vs_classification(0.01)
    plt.figure(3)
    training_size_vs_classification(0.1)
    plt.figure(4)
    training_size_vs_classification(1)
    plt.figure(5)
    training_size_vs_classification(10)
    plt.figure(6)
    training_size_vs_classification(100)  
    plt.figure(7)
    training_size_vs_classification(1000)  

def additional_features():
    plt.figure(8)
    training_size_vs_classification(10, 'b', True)
    plt.figure(9)  
    training_size_vs_classification(10, 'b',False)  

"""      
def regularization_vs_classification(train, start, stop, step):
    plt.figure(2)
    for j in range(50):
        average = {}
        
        regvalues = np.arange(start, stop, step).tolist()
        
        for trial in range(10):
            X_train,Y_train,X_test,Y_test = training_testing(X,Y,40)
            Ym = [ np.array([int(y == 1) for y in Y_train]), np.array([int(y == 2) for y in Y_train]), np.array([int(y == 3) for y in Y_train]) ]
            
            
            for i in regvalues:
                thetas = multi_normal(X_train, Ym, i)    
                results = multiclass_test(X_test, Y_test, thetas)
                if not i in average:
                    average[i] = float(sum(results))/len(results)
                else:
                    average[i] += float(sum(results))/len(results)
    
        keys = [ x[0] for x in average.items() ]
        vals = [ x[1] for x in average.items() ]
        plt.plot(regvalues, vals)
    plt.show()
"""