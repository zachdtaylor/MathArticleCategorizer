#Script for running data through Support Vector Machines Algorithm

#Imports-----------------------------------------------------------
import sys
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import time
import json
import numpy as np
import pandas as pd

#Helper Functions--------------------------------------------------
    
def train_model(X,y,model):
        model.fit(X,y)
        return model   

def runModel(data_file,model,save_model = False):
    '''
    Parameters
    ----------
    xfile = (string) name of file of all clusters representations that we have for both Math and non-Math articles
    y = (string) name of file indicating the rating of the article (0=bad, 1=good) 
    model = the type of model that you want to run [eg: for Naive Bayes => MultinomialNB()]
    '''
    
#    #we aren't using  json anymore
#    with open(xfile,'r') as f:
#        X = json.loads(f.read())
#    
#    with open(yfile,'r') as f:
#        y = json.loads(f.read())   

    # data = np.load (data_file)#this should load the file like a numpy array
    # X = data[0] #X and y need to be updated to properly index from the data file
    # y = data[0]
    
    
    #read in the data file
    data = pd.read_csv(data_file)
    X = data[data.columns[1]]
    y = data[data.columns[0]]
    
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
if __name__ == "__main__":
# Pick model by uncommenting the appropriate line #

## Naive Bayes
#model = MultinomialNB()

## Support Vector Machines
model = SVC()

print("Begin Script")
#sys.argv[0] is the name of the script
data_file = sys.argv[1]
save      = sys.argv[2]

runModel(data_file,model,save)

print("Training Complete")



