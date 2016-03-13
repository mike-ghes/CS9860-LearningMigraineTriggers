from asn1 import *
import matplotlib.pyplot as plt

class ConcreteEnum:
    Slump, Flow, CompressiveStrength = 7,8,9

def load_concrete(concEnum = ConcreteEnum.Slump):
    """
    Load the concrete dataset.
    """
    data = read_data('concrete.csv', lambda row: [float(i) for i in row[1::]], skipFirstRow = True)
    X,Y = seperate_data(data, concEnum)
    
    normalizeX = [ 1. / float(max(X[:,i])) for i in range(X.shape[1]) ]
    return X * normalizeX , Y, normalizeX


X,Y, norm = load_concrete()

X_train,Y_train,X_test,Y_test = training_testing(X,Y)

def regularization():
    thetas = np.array(normal_eqn(X_train, Y_train))
    theta2 = np.append(Jay(X_test, Y_test, thetas), Jay(X_test, Y_test, thetas))
    thetas = np.append(thetas, theta2 )
        
    for i in range(20):
        reg = 2**(i-9)
        theta = normal_eqn(X_train, Y_train, reg)
        theta2 = np.append(theta,  Jay(X_test, Y_test, theta, reg = reg) )
        theta3 = np.append(theta2,  Jay(X_test, Y_test, theta, reg = 0) )        
        thetas = np.vstack((thetas, theta3))
        
    #thetas = np.hstack(( thetas, np.array( [[ Jay(X_train, Y_train, theta) for theta in thetas ]] ) ))
    
    
    fig, ax = plt.subplots()
    plt.title('Regularization Coefficients')
    plt.xlabel('Training set size')
    plt.ylabel('Percent Misclassification')
    ax.set_yticklabels(['', '0', '2^-5', '2^0', '2^5', '2^10'])
    ax.imshow(thetas, cmap=plt.cm.gray, interpolation='nearest')

def descent_vs_normal(learningrate, reg = 0, initial = None, showJ = True):

    theta_normal = normal_eqn(X_train,Y_train, reg)
    if initial == None:
        initial = theta_0(10)
        #theta_r(len(theta_normal), max(abs(x) for x in theta_normal))
    
    theta_descent = [ initial ]
    for i in range(13):
        theta_descent += [grad_descent(X_train,Y_train, iterations = 2**i, reg = reg, rate = learningrate, theta = initial)]
        
    fig, ax = plt.subplots()
    
    
    image = np.vstack(( np.array(theta_descent), theta_normal ))
    image2 = None
    if showJ:
        
        train = np.array( [[ Jay(X_train, Y_train, image[i], reg = reg) for i in range(image.shape[0]) ]] ).T
        test = np.array(  [[ Jay(X_test,  Y_test,  image[i], reg = reg) for i in range(image.shape[0]) ]] ).T
        image2 = np.hstack((image, train, test))
    else:
        image2 = image
    
    ax.imshow(image2, cmap=plt.cm.gray, interpolation='nearest')
    
    plt.title('rate = ' + str(learningrate) + ' reg = ' + str(reg))
    fig.canvas.draw()
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[6] = '        Train/Test'
    ax.set_xticklabels(labels)
    
    labels = [''] + ["2^" + str(i*2) for i in range(7)] + ['Normal Equation']
    ax.set_yticklabels(labels)

def compare_learning_rates():
    descent_vs_normal(0.75, 4, theta_0(10), False)
    descent_vs_normal(0.5, 4, theta_0(10), False)
    descent_vs_normal(0.25, 4, theta_0(10), False)
    descent_vs_normal(0.1, 4, theta_0(10), False)
    descent_vs_normal(1, 4, theta_0(10), False)
