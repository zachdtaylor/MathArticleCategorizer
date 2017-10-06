#Script for running data through Support Vector Machines Algorithm

#Imports-----------------------------------------------------------
import sys
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import time
import json


#Helper Functions--------------------------------------------------
    
def train_model(X,y,model):
        model.fit(X,y)
        return model   

def runModel(xfile,yfile,model,save_model = False):
    '''
    Parameters
    ----------
    xfile = (string) name of file of all clusters that we have for both Math and non-Math articles
    y = (string) name of file indicating the rating of the article (0=bad, 1=good) 
    model = the type of model that you want to run [ie: for Naive Bayes => MultinomialNB()]
    '''
    
#    with open('DataSetX.txt','r') as f:
#        X = json.loads(f.read())
#    
#    with open('DataSetY.txt','r') as f:
#        y = json.loads(f.read())
    
    with open(xfile,'r') as f:
        X = json.loads(f.read())
    
    with open(yfile,'r') as f:
        y = json.loads(f.read())    
    
    #split up train/test data
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=10)
      
    #fit the model
    train_start = time.time()
    model = train_model(X_train, y_train,model)
    train_end = time.time()
    print('Training time:', train_end-train_start)
     
    benchmark = sum(y)/len(y) #should work if y is a vector of 0s and 1s
    score = model.score(X_test, y_test)
    print('Score:',score)  
    print('Benchmark:',  benchmark) 
    
    if save_model:
        filename = 'saved_model.sav'
        pickle.dump(model,open(filename,'wb'))
        print('Model saved to: ' + filename)

#Main--------------------------------------------------------------

# Pick model by uncommenting the appropriate line #

## Naive Bayes
#model = MultinomialNB()

## Support Vector Machines
model = SVC()

print("Begin Script")
#sys.argv[0] is the name of the script
xfile = sys.argv[1]
yfile = sys.argv[2]

runModel(xfile,yfile,model)

print("Training Complete")



