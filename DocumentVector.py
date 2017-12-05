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

def get_cluster(doc_number, doc_word_vectors, text_list, model):
    document = doc_word_vectors[doc_number]
    words = [word for word in clean_text_list(text_list[doc_number]) if word in model.wv]
    word_vecs = {tuple(key): value for (key, value) in zip(document, words)}

    #cls_alg = KMeans(n_clusters=50)
    cls_alg = DBSCAN(eps=.625, metric='cosine', algorithm='brute', min_samples=1)
    #cls_alg = MeanShift()
    clusters = cls_alg.fit(doc_word_vectors[doc_number])

    vector_clusters = defaultdict(set)
    for i in range(len(clusters.labels_)):
        label = clusters.labels_[i]
        vector_clusters[label].add(tuple(doc_word_vectors[doc_number][i]))

#    for cluster in vector_clusters.keys():
#        print("Cluster {}:".format(cluster+1))
#        for vector in vector_clusters[cluster]:
#            print(word_vecs[vector])
#        print("\n")
    print("clusters[1]: ", clusters[1])
    return vector_clusters[chooseCluster(clusters, words)]

def chooseCluster(clusters, doc_word_list):
    best_cluster = 0
    best_score = 0
    for cluster_id in clusters.labels_:
        score = tf_idf(clusters[cluster_id], doc_word_list)
        if score > best_score:
            best_cluster = cluster_id
            best_score = score
            
    return best_cluster

def tf_idf(cluster, doc_word_list):
    """Return the tf-idf score for the given cluster of words. Here, we define the tf-idf
    for a cluster of words to be the sum of the tf-idf scores for each individual word."""
    score = 0
    words = np.array(doc_word_list)
    for word in cluster:
        tf = np.sum(words == word) / len(words)
        idf = idf_dict[word] / total_words if word in idf_dict else 1
        score += tf * idf
    return score

# Main Function -------------------------------------------

def getVectors(file_name):
    print("Starting Script...")
    model = gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)
    print("Google words model loaded.")
    #text = xml2df("MathFeedsDataAll.xml")["Text"].as_matrix()
    clusters = []
    ratings = []
    text = xml2df(file_name)["Text"].as_matrix()
    docs = [np.array([model.wv[word] for word in clean_text_list(doc) if word in model.wv]) for doc in text if type(doc) == str]

    for x in range(len(text)):
        cluster = get_cluster(x,docs,text,model)
        return cluster
        ratings.append(x)
        break

    with open("DataSetX.txt", 'w') as f:
        f.write(json.dumps(clusters))

    with open("DataSetY.txt", 'w') as f:
        f.write(json.dumps(ratings))

#Run Above Functions
if __name__ == "__main__":
    getVectors(sys.argv[1])