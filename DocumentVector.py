#Imports-----------------------------------------------------------------------
import pandas as pd
import string
from collections import defaultdict
import gensim
import numpy as np

def getModelVector(document):
    '''Main function
    '''
    try:
        model = gensim.models.Word2Vec.load('SavedModel')
    except:
        print("Saved Model Information Not Found")
        return

def get_clusters(doc_number, num_clusters, doc_word_vectors, text_list, model):
    '''Gets a dictionary mapping cluster number to a set of vectors belonging
    to that cluster.

    Parameters
    ----------
    doc_number : int
        The index of the document in text_list
    num_clusters : int
        Number of clusters to create
    doc_word_vectors : list, contains numpy arrays
        Each position corresponds to a document, and conatins a 2D numpy
        array of shape (n_words, n_features)
    text_list : list, contains strings
        List of documents as text, where each document is a single string. Should
        be in 1-1 correspondence with doc_word_vectors.
    model : Word2Vec model
        The word2vec model that gensim returns

    Returns
    -------
    vector_clusters : dict, maps ints to sets
        Maps cluster number to a set of vectors in that cluster. The vectors
        will be the same ones as found in doc_word_vectors[doc_number], but now
        grouped into the given amount (num_clusters) of clusters.

    '''

    document = doc_word_vectors[doc_number]
    words = [word for word in clean_text_list(text_list[doc_number]) if word in model.wv]
    word_vecs = {tuple(key): value for (key, value) in zip(document, words)}

    clusters = KMeans(n_clusters=num_clusters).fit(doc_word_vectors[doc_number])

    vector_clusters = defaultdict(set)
    for i in range(len(clusters.labels_)):
        label = clusters.labels_[i]
        vector_clusters[label].add(tuple(doc_word_vectors[doc_number][i]))

    # This is just printing the words in each cluster
    for cluster in vector_clusters.keys():
        print("Cluster {}:".format(cluster+1))
        for vector in vector_clusters[cluster]:
            print(word_vecs[vector])
        print("\n")

    return vector_clusters
