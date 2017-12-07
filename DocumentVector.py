#Imports-----------------------------------------------------------------------
import pandas as pd
import string
from collections import defaultdict
import gensim
import numpy as np
import xml.etree.ElementTree as ET
from lxml import etree
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
from sklearn.cluster import KMeans
import json
import sys

#Dictionary for TF-IDF scores--------------------------------------------------
idf_dict = dict()
with open("frequentwords.txt", 'r') as f:
    for line in f:
        word, freq = line.split()
        idf_dict[word] = int(freq)

total_words = sum(idf_dict.values())

#Helper Functions--------------------------------------------------------------
def xml2df(xml_data):
    tree = ET.parse(xml_data)
    root = tree.getroot()
    all_records = []
    headers = []
    for i, child in enumerate(root):
        record = []
        for subchild in child:
            record.append(subchild.text)
            if subchild.tag not in headers:
                headers.append(subchild.tag)
        all_records.append(record)
    return pd.DataFrame(all_records, columns=headers)

def clean_word(w):
    strip_str = "()\".?!,;"
    new_word = "".join((c for c in w if c in string.printable))
    return new_word.strip(strip_str).lower()

def clean_text_list(doc):
    words = doc.split()
    clean_words = [clean_word(word) for word in words]
    return clean_words

def get_cluster(vectors, words, model):
    """
    Parameters:
        vectors (ndarray): shape is n x 300, where n is the number of words in the document
        words (list of strings): words in the document
        model (gensim.model): the model containing the Google word vectors
    """
    # Dictionary that maps each word vector to the word it represents
    word_vecs = {tuple(vector): word for (vector, word) in zip(vectors, words)}

    #cls_alg = KMeans(n_clusters=50)
    cls_alg = DBSCAN(eps=.625, metric='cosine', algorithm='brute', min_samples=1)
    #cls_alg = MeanShift()
    cluster_model = cls_alg.fit(vectors)
    return cluster_model

    # cluster_vectors is a dictionary that maps each cluster label to the set of
    # word vectors assigned to that cluster
    # cluster_words is a dictionary that maps each cluster label to the set of
    # words as strings assigned to that cluster
    cluster_vectors = defaultdict(set)
    cluster_words = defaultdict(set)
    for i in range(len(cluster_model.labels_)):
        label = cluster_model.labels_[i]
        vector = tuple(cluster_model.components_[i])
        cluster_vectors[label].add(vector)
        cluster_words[label].add(word_vecs[vector])
    
    return cluster_vectors[chooseCluster(cluster_words)]

def chooseCluster(cluster_words):
    """
    Parameters:
        cluster_words (dict): maps each cluster ID to a set of words assigned to that cluster
    Returns:
        int: the ID of the most relevant cluster
    """
    best_cluster = 0
    best_score = 0
    for Id in cluster_words:
        score = tf_idf(words[Id])
        if score > best_score:
            best_cluster = Id
            best_score = score
    return best_cluster

def tf_idf(words):
    """Return the tf-idf score for the given cluster of words.
    Parameters:
        words (set): the set of words
    Returns:
        float: the tf-idf score for the set of words

    Notes:
        Here, we define the tf-idf for a cluster of words to be the sum of the tf-idf scores 
        for each individual word.
    """
    score = 0
    words = np.array(words)
    for word in words:
        tf = np.sum(words == word) / len(words)
        idf = idf_dict[word] / total_words if word in idf_dict else 1
        score += tf * idf
    return score

# Main Function -------------------------------------------

def getVectors(file_name):
    print("Starting Script...")
    model = gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)
    print("Google words model loaded.")
    clusters = []
    ratings = []
    text = xml2df(file_name)["Text"].as_matrix()

    # docs_words contains each document as a list of words
    docs_words = [[word for word in clean_text_list(doc) if word in model.wv] for doc in text if type(doc) == str]
    # docs_vectors contains each document as an array of word vectors
    docs_vectors = [np.array([model.wv[word] for word in doc]) for doc in docs_words]

    for k in range(len(text)):
        cluster = get_cluster(docs_vectors[k],docs_words[k],model)
        return cluster
        ratings.append(k)
        break

    with open("DataSetX.txt", 'w') as f:
        f.write(json.dumps(clusters))

    with open("DataSetY.txt", 'w') as f:
        f.write(json.dumps(ratings))

#Run Above Functions
# if __name__ == "__main__":
#     getVectors(sys.argv[1])