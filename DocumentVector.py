#Imports-----------------------------------------------------------------------
import pandas as pd
import string
from collections import defaultdict
import gensim
import numpy as np

model = gensim.models.Word2Vec.load('SavedModel')
print(model.wv.most_similar(positive=['my']))
