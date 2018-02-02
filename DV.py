import gensim
import numpy as np
import string
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
from sklearn.cluster import KMeans
from collections import defaultdict

idf_dict = dict()
with open("./MathArticleCategorizer/frequentwords.txt", 'r') as f:
    for line in f:
        word, freq = line.split()
        idf_dict[word] = int(freq)

total_words = sum(idf_dict.values())

def clean_word(w):
    strip_str = "()\".?!,;"
    new_word = "".join((c for c in w if c in string.printable))
    return new_word.strip(strip_str).lower()

def clean_text_list(doc):
    words = doc.split()
    clean_words = [clean_word(word) for word in words]
    return clean_words

def doc2vec(document, model):
    words = np.array([word for word in clean_text_list(document) if word in model.wv])
    vectors = np.array([model.wv[word] for word in words])
    cluster = get_cluster(vectors, words, model)
    return np.sum(np.asarray(list(cluster)), axis=0) / len(cluster)

def get_cluster(vectors, words, model):
    """
    Parameters:
        vectors (ndarray): shape is n x 300, where n is the number of words in the document
        words (ndarray): the entire document as a list of words (not just the distinct words)
        model (gensim.model): the model containing the Google word vectors
    """
    # Dictionary that maps each word vector to the word it represents
    word_vecs = {tuple(vector): word for (vector, word) in zip(vectors, words)}

    #cls_alg = KMeans(n_clusters=50)
    cls_alg = DBSCAN(eps=.625, metric='cosine', algorithm='brute', min_samples=1)
    #cls_alg = MeanShift()
    cluster_model = cls_alg.fit(vectors)

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
    idx = chooseCluster(cluster_words, words)
    return cluster_vectors[idx]

def chooseCluster(cluster_words, words):
    """
    Parameters:
        cluster_words (dict): maps each cluster ID to a set of words assigned to that cluster
        words (ndarray): the entire document as a list of words
    Returns:
        int: the ID of the most relevant cluster
    """
    best_cluster_ID = 0
    best_score = 0
    for Id in cluster_words:
        score = tf_idf(cluster_words[Id], words)
        if score > best_score:
            best_cluster_ID = Id
            best_score = score
    return best_cluster_ID

def tf_idf(cluster, words):
    """Return the tf-idf score for the given cluster of words.
    Parameters:
        cluster (set): the set of words in the cluster
        words (ndarray): the entire document as a list of words
    Returns:
        float: the tf-idf score for the set of words

    Notes:
        Here, we define the tf-idf for a cluster of words to be the sum of the tf-idf scores 
        for each individual word.
    """
    score = 0
    for word in cluster:
        tf = np.sum(words == word) / len(words)
        idf = idf_dict[word] / total_words if word in idf_dict else 1
        score += tf * idf
    return score