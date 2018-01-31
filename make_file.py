#==============================================================================
# make_file.py
# runs doc2vec function in DV.py for each article in the corpus
# saves the resulting score for each document, and the document's class 
# to a file.
#==============================================================================

#Imports-----------------------------------------------------------------------
import DV
# import pickle
import pandas as pd
import gensim
import sys

#Helper Functions--------------------------------------------------------------
def cleanArticles(df):
    '''
    Removes duplicate articles and blank articles from df
    
    Parameters
    ----------
    df = pandas data frame containing only text for each document
    '''
    clean_df = df.drop_duplicates() #drop duplicates
    blank_index = clean_df.index[clean_df.iloc[:,0]==''].tolist()#index of blank article
    clean_df = clean_df.drop(blank_index,axis=0) #drop blanks
    return(clean_df)
    
def getValues(df, model):
    '''
    Loops through all documents and stores score to a list.
    The list of scores is then returned.
    
    Parameters
    ----------
    df = 'cleaned' data frame from 'cleanArticles' function
    model = model used for doc2vec (e.g. google's news model)
    '''
    values = []
    for i in range(df.shape[0]):
        doc_score = DV.doc2vec(df.iloc[i,0],model)
        values.append(doc_score)
    return(values)

def createFile(good_scores, bad_scores, filename = 'algorithm_file.csv', return_df = False):
    '''
    Merges lists of good and bad scores into a pandas data frame, and adds 
    column for class of article (good ==1 or bad == 0)
    Writes the pandas df to a file and returns the data.frame if specified
    
    Parameters
    ----------
    good(bad)_scores = list of scores from 'getValues' function
    filename = string of what to name the file to which the data are saved
    return_df = boolean indicating if df should be returned
    '''
    #make class labels for good/bad articles
    bad_class = [0 for i in range(len(bad_scores))]
    good_class = [1 for i in range(len(good_scores))]
   
    # combine and create pandas df
    all_values = bad_scores + good_scores
    all_classes = bad_class + good_class
    combined_data = pd.DataFrame({
    'Value': all_values,
    'Class': all_classes
    })
    
    #save data to file that can be fed into the algorithm
    combined_data.to_csv(filename,index = False)
    
    if (return_df == True):
        return combined_data
    else:
        return None
    

    
    
# Main Function ---------------------------------------------------------------
def runScript(good_file,bad_file):
    
    print("Starting 'make_file.py'... ")
       
    #run google model
    print('1/4: Initializing google model')
    google_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews.bin', binary=True)
    
    #clean up article data
    good_file = good_file.to_frame()#since good articles are series instead of dataframe...
    print('2/4: Cleaning Data')
    good_clean = cleanArticles(good_file)
    bad_clean  = cleanArticles(bad_file)
    
    #get score values
    print("3/4: Getting Scores")
    good_scores = getValues(good_clean,google_model)
    bad_scores  = getValues(bad_clean,google_model)
    
    #write scores and classes to a file
    print("4/4 Writing to file")
    createFile(good_scores,bad_scores)    
    
    print("Done")
    
    return(None)
    
    
    
    


#Run Above Functions
if __name__ == "__main__":
    #select good and bad article files
#    good_articles = pd.read_pickle("GoodArticles.pkl")
#    good_articles = good_articles.to_frame() #may not be necessary to do this
    good_articles = pd.read_pickle(sys.argv[1])
    
#    bad_articles=pd.read_pickle("BadArticles.pkl")
    bad_articles=pd.read_pickle(sys.argv[2])
    
    
    #sys.argv[1] and sys.argv[2] are good and bad article files respectively
#    runScript(sys.argv[1],sys.argv[2])
    runScript(good_articles,bad_articles)
















