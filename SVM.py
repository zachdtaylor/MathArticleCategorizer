#Script for running data through Support Vector Machines Algorithm

#Imports-----------------------------------------------------------

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import time
import json


#Helper Functions--------------------------------------------------
def train_svm(X, y):
    svm = SVC()
    svm.fit(X,y)
    return svm 



#Main--------------------------------------------------------------

def runSVC(xfile,yfile):
    '''
    Parameters
    ----------
    xfile = File of all of the clusters that we have for both Math and non-Math articles
    y = file indicating the rating of the article (0=bad, 1=good)   
    '''
    
    with open('DataSetX.txt','r') as f:
        X = json.loads(f.read())
    
    with open('DataSetY.txt','r') as f:
        y = json.loads(f.read())
       
    #split up train/test data
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=10)
      
    #fit the model
    train_start = time.time()
    svm = train_svm(X_train, y_train)
    train_end = time.time()
    print('Training time:', train_end-train_start)
     
    benchmark = sum(y)/len(y) #should work if y is a vector of 0s and 1s
    score = svm.score(X_test, y_test)
    print('Score:',score)  
    print('Benchmark:',  benchmark) 

#run the function
runSVC(xfile,yfile)



