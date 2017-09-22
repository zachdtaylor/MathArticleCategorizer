#Script for running data through Support Vector Machines Algorithm

#Imports-----------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import time



#Helper Functions--------------------------------------------------
def train_svm(X, y):
    svm = SVC()
    svm.fit(X,y)
    return svm 



#Main--------------------------------------------------------------


def runSVC(X,y):
    '''
    Parameters
    ----------
    X = All of the clusters that we have for both Math and non-Math articles
    y = Whether or not  the articles associated with the clusters are 'high-quality'

    
    '''
#split up train/test data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=10)


#fit the model
train_start = time.time()
svm = train_svm(X_train, y_train)
train_end = time.time()
print('Training time:', train_end-train_start)
 
benchmark = 
score = svm.score(X_test, y_test)
print('Benchmark: )
print('Score:',score)




