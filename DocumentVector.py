#Imports-----------------------------------------------------------------------
import pandas as pd
import string
from collections import defaultdict
import gensim
import numpy as np

#Load Model from file Saved Model
def getModelVector(document):
    try:
        model = gensim.models.Word2Vec.load('SavedModel')
    except:
        print("Saved Model Information Not Found")
        return

    
