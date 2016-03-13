from asn1 import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import datetime
import copy
from math import isnan


from sklearn import feature_selection
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier#Best
from sklearn.naive_bayes import GaussianNB
from sklearn import ensemble
from sklearn import linear_model
from sklearn import neighbors
from sklearn import tree




features = ["Bias","Cold air exposure","Nightshade","Odors","Physical exertion","Overslept","Lack of sleep","Post-stress","Stress","Missed a meal",
    "Smoked or cured meat","Bananas","Caffeine","Citrus","Beer","Cheese","Chocolate","Red wine","Lights","Spirits","Loud sounds","Sugar","Dehydration",
    "Changing weather","Hot and humid weather"]

food    = ["Nightshade", "Smoked or cured meat", "Bananas", "Cheese", "Chocolate", "Sugar", "Missed a meal"]
drink   = ["Caffeine", "Beer", "Red wine", "Spirits",  "Citrus", "Dehydration"]
ambient = ["Cold air exposure", "Odors", "Lights", "Loud sounds", "Changing weather", "Hot and humid weather"]
physical= ["Physical exertion", "Overslept", "Lack of sleep", "Post-stress", "Stress"]

day     = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
month   = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

behavioural = ["Bias"] + food + drink + ["Cold air exposure", "Loud sounds", "Physical exertion", "Overslept", "Lack of sleep"]
allfeat = features + day + month
nofeat  = []
    
    
def load_migraine(mask = allfeat):
    """
    Load migraine dataset.
    """
    def selector(row):
        date_str = row[0]
        date = datetime.datetime(int(date_str[6:10]), int(date_str[3:5]), int(date_str[0:2]))
        return [int(i) for i in ([row[1]] + row[3::])] + [int(i == date.weekday()) for i in range(7)] + [int(i == date.month) for i in range(12)]
    
    data = read_data('migraine_data_inverted.csv', selector, True ) # get rid of date [0] and comment [2]. Convert everything else to int.
     
    X,Y = seperate_data(data, 0) # output is stored at index 0.
    
    indices = [ allfeat.index(v) for v in mask ]
    
    X = X[:, indices]
    features = np.array(allfeat)[:, indices].tolist()
    return X, Y, features

def load_migraine_past(past_days, mask = allfeat):
    """
    Load migraine dataset using offset days.
    :param past_days: Use these offset days. e.g. [0] is identical to load_migraine(), [1,2,3] uses past 3 days (but not today)
    """
    X,Y, key = load_migraine(mask)

    dupmask = [ x in day+month+["Bias"] for x in key] #Mask of features to duplicate 
    ndupmask = [ not x for x in dupmask ] # Negative mask of features to duplicate
    
    Ynew =   Y[max(past_days):]
    Xnew =   X[max(past_days):].compress(dupmask, 1) # Extract day/month/bias since we don't want them copied.
    
    X_rest = X.compress(ndupmask, 1) 
    
    print Xnew.shape, X_rest.shape  
    keynew = [ key[i] for i in range(len(key)) if dupmask[i]]
    key_rest = [ key[i] for i in range(len(key)) if ndupmask[i]]
    print keynew, key_rest  
    
    for offset in past_days:
        start = max(past_days)-offset
        end = X_rest.shape[0]-offset
        prev_day = X_rest[start:end]
        Xnew = np.hstack((Xnew, prev_day))
        keynew += [key + str(offset) for key in key_rest]
            
    return Xnew, Ynew, keynew 
################################################################################
# Using code from assignment 1. Non-optimized and possibly(?) buggy.
# Initial analysis done with this was discarded in favour of using sklearn functions.
################################################################################

def migraine_test(X2, Y2, theta, thresh = 0.41, aspercent = True):
    incr = 100./len(X2) if aspercent else 1
    
    results = [[0, 0], [0, 0]]
    for i in range(len(X2)):
        h0 = H_log(theta, X2[i])        
        results[h0< thresh][Y2[i]] += incr 
    
    return results

def weighted_migraine_test(X2, Y2, theta, aspercent = True):
    incr = 100./len(X2) if aspercent else 1
    
    results = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(len(X2)):
        h0 = H_lin(theta, X2[i])
        if h0 < 0.6:     
            results[0][Y2[i]] += incr 
        elif h0 > 0.95:
            results[2][Y2[i]] += incr
        else:
            results[1][Y2[i]] += incr  
    return results


def find_triggers(theta, negthresh = -0.1, posthresh = 0.1):
    badindx = [ i for i in range(len(theta)) if theta[i] < negthresh ]
    print "bad:  ", [key[i] + "(" + str(theta[i]) + ")" for i in badindx]
    goodindx = [ i for i in range(len(theta)) if theta[i] > posthresh ]
    print "good: ", [ key[i] + "(" + str(theta[i]) + ")" for i in goodindx]

def run_naive(X, Y, trainingSz = 400, reg = 10):
    X_train,Y_train,X_test,Y_test = training_testing(X,Y, trainingSz)

    Y_test_mild = np.array([int(y >= 1) for y in Y_test])
    Y_test_sevr = np.array([int(y >= 2) for y in Y_test])
    Y_train_mild = np.array([int(y >= 1) for y in Y_train])
    Y_train_sevr = np.array([int(y >= 2) for y in Y_train])

    theta_mild = normal_eqn(X_train,Y_train_mild, reg = reg)
    theta_sevr = normal_eqn(X_train,Y_train_sevr, reg = reg)
    theta = normal_eqn(X_train, Y_train, reg = reg)
    
    # Training
    ax = plt.subplot(2,3,1)
    ax.imshow(migraine_test(X_train, Y_train_mild, theta_mild), interpolation='nearest')
    ax.set_title("Train - Mild")
    
    ax2 = plt.subplot(2,3,2)
    ax2.imshow(migraine_test(X_train, Y_train_sevr, theta_sevr), interpolation='nearest')
    ax2.set_title("Train - Severe")
    
    ax3 = plt.subplot (2,3,3)
    ax3.imshow(weighted_migraine_test(X_train, Y_train, theta), interpolation='nearest')
    ax3.set_title("Train - Weighted")
    
    # Testing
    ax4 = plt.subplot(2,3,4)
    ax4.imshow(migraine_test(X_test, Y_test_mild, theta_mild), interpolation='nearest')
    ax4.set_title("Test - Mild")
    
    ax5 = plt.subplot(2,3,5)
    ax5.imshow(migraine_test(X_test, Y_test_sevr, theta_sevr), interpolation='nearest')
    ax5.set_title("Test - Severe")
    
    ax6 = plt.subplot (2,3,6)
    ax6.imshow(weighted_migraine_test(X_test, Y_test, theta), interpolation='nearest')
    ax6.set_title("Test - Weighted")
    
    plt.show()
    print "\ntrain"
    print "mild:     ", migraine_test(X_train, Y_train_mild, theta_mild), "[[true negative, false negative], [false positive, true positive]]"
    print "severe:   ", migraine_test(X_train, Y_train_sevr, theta_sevr), "[[true negative, false negative], [false positive, true positive]]"
    print "weighted: ", weighted_migraine_test(X_train, Y_train, theta) 
    print "\ntest"
    print "mild:     ", migraine_test(X_test, Y_test_mild, theta_mild), "[[true negative, false negative], [false positive, true positive]]"
    print "severe:   ", migraine_test(X_test, Y_test_sevr, theta_sevr), "[[true negative, false negative], [false positive, true positive]]"
    print "weighted: ", weighted_migraine_test(X_test, Y_test, theta) 

    print "\nSevere triggers:"
    find_triggers(theta_sevr, -0.07, 0.07)
    print "\nMild triggers:"
    find_triggers(theta_mild)

    return theta, theta_mild, theta_sevr

################################################################################
# Comparison of sklearn classifiers. Plotted in confusion matrices.
# I eventually switched over to LOOCV instead of a test/training split
################################################################################
def run_sk(trainingSz = 40):    
    classifiers = [ ("Linear SVC", svm.LinearSVC()), 
                    ("Linear weighted SVC", svm.LinearSVC(class_weight={0:1, 1:3 , 2:6 })), 
                    ("rbf SVC", svm.SVC(kernel='rbf')),
                    ("rbf weighted SVC", svm.SVC(kernel='rbf', class_weight={0:1, 1:3 , 2:6 })),
                    ("Linear kernal SVC", svm.SVC(kernel='linear')),
                    ("Linear weighted kernal SVC", svm.SVC(kernel='linear', class_weight={0:1, 1:3 , 2:6 })),
                    ("Poly kernal SVC", svm.SVC(kernel='poly')),
                    ("Poly weighted kernal SVC", svm.SVC(kernel='poly', class_weight={0:1, 1:3 , 2:6 })),               
                    #("Sigmoid kernal SVC", svm.SVC(kernel='sigmoid')),
                    ("SVM Regression", svm.SVR()),             
                    #("Adaboost", AdaBoostClassifier(n_estimators=100)),
                    ("Gradient Boost", GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)),             
                    ("Decision Tree", DecisionTreeClassifier(max_depth=5)),                   
                    ("Gaussian", GaussianNB()), 
                    ("Random Tree10", RandomForestClassifier(n_estimators=10)),
                    ("Random Tree5", RandomForestClassifier(n_estimators=5)),
                    ("Random Tree20", RandomForestClassifier(n_estimators=20)),
                    ("Random Tree50", RandomForestClassifier(n_estimators=50)),
                  ]
    
    j = 0  
    
    # Preprocessing
    
    #X_scaled = preprocessing.scale(X)
    
    #pca = PCA(n_components=10)
    #X_r = pca.fit(X).transform(X)

    #lda = LDA(n_components=2)
    #X_r2 = lda.fit(X, Y).transform(X)
    
    X_train,Y_train,X_test,Y_test = training_testing(X,Y, trainingSz)  
    X_train_e,Y_train_e,X_test_e,Y_test_e = training_testing_equalized(X,Y, trainingSz)
                                                
    for j, (name, clf) in enumerate(classifiers):
        j+=1
        for k, (X_train, Y_train, X_test, Y_test) in enumerate([training_testing(X,Y, trainingSz), training_testing_equalized(X,Y, trainingSz) ]):
        
                    
            clf.fit(X_train, Y_train)
            
            
            results = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            for i in range(len(X_train)):
                h0 = clf.predict(X_train[i])
                results[h0][Y_train[i]] += 1 
    
            ax = plt.subplot(4, len(classifiers), j + (2*k)*len(classifiers) )
            ax.imshow(results, interpolation='nearest')
            ax.set_title(name + " - Train")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
                    
                                    
            results2 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            for i in range(len(X_test)):
                h0 = clf.predict(X_test[i])
                results2[h0][Y_test[i]] += 1 
    
            ax2 = plt.subplot(4, len(classifiers), j + ((2*k)+1)*len(classifiers))
            ax2.imshow(results2, interpolation='nearest')
            ax2.set_title(name + " - Test")
            ax2.set_xlabel("Actual")
            ax2.set_ylabel("Predicted")
            
            
            """
            results3 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            for i in range(len(X_scaled)):
                h0 = clf.predict(X_scaled[i])
                results3[h0][Y[i]] += 1
            ax3 = plt.subplot(3, len(classifiers), j + 2*len(classifiers))
            ax3.imshow(results3, cmap=plt.cm.gray, interpolation='nearest')
            ax3.set_title(name + " - Combined")
            """
        
    plt.show()


###############################################################################
# Switching over to LOOCV. Code used for many of the diagrams from presentation
###############################################################################

def LOOCV_homebrew(X,Y, clf = None, reg = 10):
    results1, results2, results3 = [0,1,3], [0,1,3], [0,1,3]           
    if clf == None:
        from sklearn.linear_model import SGDClassifier
        clf = 'SGDClassifier(loss="log", n_iter= 1000)'
    
    for i in range(len(Y)):        
        X_removed = np.vstack((X[:i], X[i+1:]))
        Y_removed = np.hstack((Y[:i], Y[i+1:]))
        clf1 = eval(clf).fit(X_removed, [int(y >= 1) for y in Y_removed])
        clf2 = eval(clf).fit(X_removed, [int(y >= 2) for y in Y_removed])
        
        print Y[i], clf1.predict(X[i]), clf2.predict(X[i])
        predict1 = clf1.predict(X[i])[0]
        predict2 = clf2.predict(X[i])[0]
        
        if Y[i] == 0:
            results1 += [ predict1 + 2*predict2]#+ predict2 ]
            #print Y[i], clf.predict(X[i]), clf.predict_proba(X[i])
        elif Y[i] == 1:
            results2 += [ predict1 + 2*predict2]
            #print Y[i], clf.predict(X[i]), clf.predict_proba(X[i])
        else:
            results3 += [ predict1 + 2*predict2]
    
    
    plt.hist(results1, alpha=0.5, color='g')  
    plt.hist(results2, alpha=0.5, color='y')               
    plt.hist(results3, alpha=0.5, color='r')
        
    return results1, results2, results3

def LOOCV_comparePredicts(X,Y):
    fig_index = 0
    

    clfs = [
            #Linear models:
            #("Linear Regression", linear_model.LinearRegression(normalize = False)),
            ("Linear Regression Normalize", linear_model.LinearRegression(normalize = True)),       #Does nothing
            #]+[("Linear Ridge alpha = " + str(a), linear_model.Ridge (alpha = a)) for a in np.arange(0.1,0.9,0.1) ]+[
            #("Lasso", linear_model.Lasso(alpha = 0.1)),            #Garbage
            #("LARS", linear_model.Lars()),
            #("LARS Lasso", linear_model.LassoLars(alpha=.1)),      #Garbage
            #("OMP", linear_model.OrthogonalMatchingPursuit()),     #Garbage
            
            #SVR kernals:
            ("SVR - rbf", svm.SVR(kernel='rbf')),                  #Not bad
            ("SVR - lin", svm.SVR(kernel='linear')),               #Not bad
            ]+[("SVR - poly" +str(deg), svm.SVR(kernel='poly', degree = deg)) for deg in [3] ]+[                 #Best kernal
            
            #knn
            ]+[ ("knn - uniform k=" + str(k), neighbors.KNeighborsRegressor(k, weights='uniform')) for k in [
            #        5,10,15,
                    
                    20,
                    
            #        25,30,40,50,60,80, 100, 125, 150, 200 
                    ] ]+[
            #]+[ ("knn - uniform k=" + str(k), neighbors.KNeighborsRegressor(k, weights='distance')) for k in [5,10,15,20,25,30,50,75,100] ]+[      #BROKEN
            
            
            #Decision tree
            #("Decision Tree", tree.DecisionTreeRegressor()),   #broken?
            
          ]
          
    
    for clf in clfs:
        print clfs
        fig_index+=1
        plt.figure(fig_index)
        #LOOCV_predict(X,Y,clf[1], copy.deepcopy(clf[1]), clf[0])       
        #LOOCV_predict(X,Y,clf[1], None, clf[0] + " - Severe only")   
        LOOCV_predict(X,Y,clf[1], None, clf[0] + " - Mild only") 
    



def LOOCV_predict(X,Y,clf1, clf2,name):
    results1, results2, results3 = [0],[0],[0]
    
    for i in range(len(Y)):        
        
        X_removed = np.vstack((X[:i], X[i+1:]))
        Y_removed = np.hstack((Y[:i], Y[i+1:]))
        if clf1 != None:
            clf1 = clf1.fit(X_removed, [y if y<=1 else 2 for y in Y_removed])
        if clf2 != None:
            clf2 = clf2.fit(X_removed, [int(y >= 2) for y in Y_removed])
        
        
 
        predict1 = 0 if clf1 == None else clf1.predict(X[i])
        predict2 = 0 if clf2 == None else clf2.predict(X[i])
        
        print Y[i], predict1, predict2
        
        if Y[i] == 0:
            results1 += [ predict1 + 2*predict2 ]
            #print Y[i], clf.predict(X[i]), clf.predict_proba(X[i])
        elif Y[i] == 1:
            results2 += [ predict1 + 2*predict2 ]
            #print Y[i], clf.predict(X[i]), clf.predict_proba(X[i])
        else:
            results3 += [ predict1 + 2*predict2 ]
    
    ax = plt.subplot()
    ax.set_title(name)
    ax = plt.hist(results1, alpha=0.5, bins=40, color='g')  
    ax += plt.hist(results2, alpha=0.5, bins=40, color='y')               
    ax += plt.hist(results3, alpha=0.5, bins=40, color='r')
    
    
    return results1, results2, results3


def LOOCV_3D(X,Y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('"No" probability')
    ax.set_ylabel('Mild probability')
    ax.set_zlabel('Severe probability')
    for i in range(len(Y)):    
        X_removed = np.vstack((X[:i], X[i+1:]))
        Y_removed = np.hstack((Y[:i], Y[i+1:]))
        
        clf1 = svm.SVC(kernel="poly", degree = 3)
        #clf2 = linear_model.LinearRegression()
        clf1.fit(X_removed, Y_removed)
        #clf1.fit(X_removed, [int(y >= 1) for y in Y_removed])
        #clf2.fit(X_removed, [int(y >= 2) for y in Y_removed])
        predict1 = clf1.predict_proba(X[i])
        #predict2 = clf2.predict(X[i])
        
        print predict1[0]#, predict2
        
        c = 'g'
        if Y[i] == 2:
            c = 'r'
        elif Y[i] == 1:
            c = 'y'
        
        ax.scatter(predict1[0][0], predict1[0][1], predict1[0][2], c=c)


################################################################################
# Finally getting to code used for the paper.
################################################################################


def compare_models(X,Y):
    ax = plt.gca()
    
    outstr = []
    
    clfs = [
            linear_model.LinearRegression(normalize = True),
            neighbors.KNeighborsRegressor(40, weights='uniform', warn_on_equidistant=False),
            neighbors.KNeighborsRegressor(60, weights='uniform', warn_on_equidistant=False),
            neighbors.KNeighborsRegressor(80, weights='uniform', warn_on_equidistant=False),
            svm.SVR(kernel='linear'),
            svm.SVR(kernel='poly', degree = 2),
            svm.SVR(kernel='poly', degree = 3),
            svm.SVR(kernel='rbf'),
            linear_model.Ridge(),
            linear_model.Lars(),
            linear_model.OrthogonalMatchingPursuit(),
            tree.DecisionTreeRegressor()
            ]
    ax.plot(-1,0)
    
    maxJ = len(clfs)
    ax.plot(maxJ+1, 0)
    
    optimal = -float('inf')
    for j in range(maxJ):
        print j
        
        #clf =  svm.SVR(kernel='poly', degree = j)
        #clf =  neighbors.KNeighborsRegressor(j, weights='uniform', warn_on_equidistant=False)
        clf =  clfs[j]
        #clf =  svm.SVR(kernel='poly', degree = j)
        
        
        predict = []
        for i in range(len(Y)):
            X_removed = np.vstack((X[:i], X[i+1:]))
            Y_removed = np.hstack((Y[:i], Y[i+1:]))
            Y_transformed = [0 if y==0 else 
                             1 if y ==2 else 
                             0.5
                             for y in Y_removed]
            clf = clf.fit(X_removed, Y_transformed)
            predict += [clf.predict(X[i])]
                 
        y0 = [predict[i] for i in range(len(predict)) if Y[i]==0]
        y1 = [predict[i] for i in range(len(predict)) if Y[i]==1]
        y2 =[predict[i] for i in range(len(predict)) if Y[i]==2]
        
        ax.errorbar(j, np.mean(y0), yerr=np.vstack([np.std(y0), np.std(y0)]), color='g')
        ax.errorbar(j, np.mean(y1), yerr=np.vstack([np.std(y1), np.std(y1)]), color='b')
        ax.errorbar(j, np.mean(y2), yerr=np.vstack([np.std(y2), np.std(y2)]), color='r')
        
        outstr+= [str(clf)]
        outstr+= ["no     - mean: " + str(np.mean(y0)) + " std: " + str(np.std(y0)) ]
        outstr+= ["mild   - mean: " + str(np.mean(y1)) + " std: " + str(np.std(y1)) ]
        outstr+= ["severe - mean: " + str(np.mean(y2)) + " std: " + str(np.std(y2)) ]
        
        diffmean = (np.mean(y0) - np.mean(y1))**2 + (np.mean(y1) - np.mean(y2))**2 + abs(np.mean(y0)- np.mean(y2))**2
        sumstd = np.std(y0)**2 + np.std(y1)**2 + np.std(y2)**2
        if np.mean(y0) == 0:
            diffmean = abs(np.mean(y1) - np.mean(y2))
            sumstd = np.std(y1) + np.std(y2)
        outstr+= [ diffmean]
        outstr+= [ sumstd ]       
        #ax.draw()
        optimal = max(optimal, diffmean - sumstd)
        ax.plot(j, np.mean(y0), 'go')
        ax.plot(j, np.mean(y1), 'bo')
        ax.plot(j, np.mean(y2), 'ro')
        
        ax.plot(j, diffmean - sumstd, 'ko')
    ax.axhline(y=optimal, color='r')
    return outstr
        
        
    


def feature_select(X,Y,key):
    
    models = [
                feature_selection.SelectPercentile(),
                feature_selection.SelectKBest(),
                feature_selection.SelectKBest(k=5),
                feature_selection.SelectFpr(),
                feature_selection.SelectFdr(),
                feature_selection.SelectFwe()
                #feature_selection.RFE()
            ]
    for select in models:
        select.fit(X,Y)
        print select, select.transform(key)
        zipped = sorted([[key[i], select.scores_[i]] for i in range(len(key))], key=lambda x: -x[1])
        zipped2 = sorted([[key[i], select.pvalues_[i]] for i in range(len(key))], key=lambda x: x[1])
    return zipped, zipped2
    
def features_thenReFit(X,Y, key):
    z_score, z_pval = feature_select(X,Y, key)
    newmask= [ x[0] for x in z_score ][:20]
    X2 = X[:,[i for i in range(len(key)) if key[i] in newmask]]
    optimize_theta(X2, Y)
    print newmask

    
def feature_from_classification(X,Y,key):
    Y1 = np.array([ 1 if x >=1 else 0 for x in Y ])
    Y2 = np.array([ 1 if x >=2 else 0 for x in Y ])
    
    clf1 = linear_model.LogisticRegression(penalty="L1").fit(X,Y1)
    clf2 = linear_model.LogisticRegression(penalty="L1").fit(X,Y2)
    
    z1 = [ [ key[i], clf1.coef_[0,i] ] for i in range(len(key)) ]
    z2 = [ [ key[i], clf2.coef_[0,i] ] for i in range(len(key)) ]
    
    z3 = [ [ key[i], clf1.coef_[0,i] + clf2.coef_[0,i] ] for i in range(len(key)) ]
    
    return sorted( z3, key=lambda x: -x[1] )
    
def features_kbest(X,Y,key):
    funcs = [
                feature_selection.f_regression,
                feature_selection.chi2,
                feature_selection.f_classif
    
            ]
    for func in funcs:
        clf = feature_selection.SelectKBest(func, 10).fit(X,Y)
        z = [ [ key[i], 0 if isnan(clf.scores_[i]) else clf.scores_[i] ] for i in range(len(key)) ]
        print func
        print clf.transform(key)
        for x in sorted(z, key=lambda x: -x[1])[:10]:
            print [x[0], float("{:0.4f}".format(x[1]))]

def importance(X,Y, k):
    out = None
    for x in [k]:
        clf =feature_selection.SelectPercentile()
        clf.fit(X,Y)
        if out == None:
            out = clf.get_support().astype(int)
        else:
            out += clf.get_support().astype(int)
    return out
    
def excel_column(X,Y, clf):
    clf.fit(X,Y)
    if len(clf.coef_) <=1:
        for x in clf.coef_[0]:
            print x
    else:
        for x in clf.coef_:
            print x
    

def confusion_matrix(X,Y, clf):
    out = np.array([[0,0,0],[0,0,0],[0,0,0]])
    
    for i in range(len(Y)):    
        X_removed = np.vstack((X[:i], X[i+1:]))
        Y_removed = np.hstack((Y[:i], Y[i+1:]))
        predict = clf.fit(X_removed, Y_removed).predict(X[i])
        out[Y[i]][predict] += 1
    return out
    
def confusion_matrix2(X,Y1, Y2, clf):
    out = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    
    for i in range(len(Y)):    
        X_removed = np.vstack((X[:i], X[i+1:]))
        Y1_removed = np.hstack((Y1[:i], Y1[i+1:]))
        Y2_removed = np.hstack((Y2[:i], Y2[i+1:]))
        predict1 = clf.fit(X_removed, Y1_removed).predict(X[i])
        predict2 = clf.fit(X_removed, Y2_removed).predict(X[i])
        out[Y[i]][predict1 + 2*predict2] += 1
    return out     
       
def ignore_test(m):
     new = m[:,[0,1,3]] 
     new[0,2] += m[0,2]
     return new              
        
def interpret_confusion(m):
    total = float(sum( [sum(x) for x in m] ))
    accuracy =  (m[0][0] + m[1][1] + m[2][2]) / total
    false_pos = (m[0][1] + m[0][2]) / total
    false_neg = (m[1][0] + m[2][0]) /total
    false_str = (m[1][2] + m[2][1]) / total
    return accuracy, false_pos, false_neg, false_str

def confusion_diagram(X,Y,Y1,Y2):
    clfs = [
                #svm.LinearSVC(),
                #svm.SVC(kernel="linear"),
                #svm.SVC(kernel="rbf"),
                #svm.SVC(kernel="poly", degree=3)
                neighbors.KNeighborsClassifier(10),
                neighbors.KNeighborsClassifier(30),
                neighbors.KNeighborsClassifier(50)
                #linear_model.LogisticRegression(),
                #linear_model.RandomizedLogisticRegression(),
                #linear_model.PassiveAggressiveClassifier(),
                #linear_model.RidgeClassifier()
            ] 
    out = [] 
    i = 0      
    for clf in clfs:
        i+=1
        print clf, "1"
        cm1 = confusion_matrix(X,Y, clf)
        print clf, "2"
        cm2 = ignore_test(confusion_matrix2(X,Y1, Y2, clf))
        
        ax = plt.subplot(3, len(clfs), i)
        ax.imshow(cm1, interpolation='nearest')
        
        ax = plt.subplot(3, len(clfs), len(clfs)+i)
        ax.imshow(cm2, interpolation='nearest')
        out += [[clf, cm1, cm2]]
    return out
                        
def features_from_models(X,Y,key):       
    regr = [
             ensemble.GradientBoostingRegressor(),
             #ensemble.RandomForestRegressor(),
             #ensemble.ExtraTreesRegressor(),
             #ensemble.AdaBoostRegressor(),
             
             #svm.SVR(kernel='linear'),
             #tree.DecisionTreeRegressor()
            ]
    for clf in regr:
        clf.fit(X,Y)
        z = [ [key[i], clf.feature_importances_[i]] for i in range(len(key))]
        
        z = sorted(z, key=lambda x: -x[1])
        print z[:10]
    for clf in clsf:
        clf.fit(X,Y)
        z = [ [key[i], clf.feature_importances_[i]] for i in range(len(key))]
        
        z = sorted(z, key=lambda x: -x[1])
        print z[:10]
                                                         
#X,Y, key = load_migraine()

days = [1,2]
X,Y, key = load_migraine_past(days, allfeat)
Y1 = np.array([ 1 if x >=1 else 0 for x in Y ])
Y2 = np.array([ 1 if x >=2 else 0 for x in Y ])


