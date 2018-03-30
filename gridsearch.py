# This file is used to run a grid search algorithm to tune the SVM
# to run in command line use : python gridsearch.py algorithm_file.csv 0
#the file will take a moment before printing out "Starting Script..."

# Libraries
import sys
import time
#import datetime
import pandas as pd
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

#Main--------------------------------------------------------------
if __name__ == "__main__":
    # Pick model by uncommenting the appropriate line #

    print("Starting Script...")
    #sys.argv[0] is the name of the script
    data_file = sys.argv[1]
    save      = int(sys.argv[2])
    # model_name = str(sys.argv[3])
    
    #read in the data file
    data = pd.read_csv(data_file)
    X = data.drop('Class',axis = 1)
    y = data.Class
    
    #set up the grid search
    
    #parameters to tune
    k_list = ['rbf']#['linear','rbf','poly']#,'sigmoid']#,'precomputed']
    gam_list = [0.1]#,0.125]#['auto',0.01,0.1]#,.5,1,10,100]
    c_list = [220,300]#list(range(185,205,1))#[195,197,199]
    # deg_list = [1,2,3,4]#,5]

    #all together
    param_dict = dict(kernel=k_list,gamma=gam_list,C = c_list)

    #model
    model = SVC()

    grid = GridSearchCV(cv=3,estimator=model, param_grid=param_dict)
    start = time.time()
    grid.fit(X,y)
    end = time.time()
    runtime = end-start
    print('Minutes:',runtime/60)   

    print("Best Score:",grid.best_score_)
    print('Params:',grid.best_params_)
    
    if(save):
        #write paramater that were searched and the best selected and the date
        import datetime
        now = datetime.datetime.now()
        param_string = 'Params Searched:' + str(param_dict) #param dictionary
        grid_best_score = "Best Score:" + str(grid.best_score_)
        grid_best_parms = 'Best Params:' + str(grid.best_params_)
        
        filename = 'best_params.txt' #create file
        file_obj = open(filename,w)
        file_obj.write("Date/Time:" + str(now)[:-7] + '\n') #write the date
        file_obj.write(param_string + '\n')
        file_obj.write(grid_best_score + '\n')
        file_obj.write(grid_best_params + '\n')
        
        file_obj.close()
        
        print('Params written to: ' + filename)
        

    print("End Script")


### End of Script ###





















