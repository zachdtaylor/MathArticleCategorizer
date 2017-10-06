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

def get_clusters(doc_number, doc_word_vectors, text_list, model):
    document = doc_word_vectors[doc_number]
    words = [word for word in clean_text_list(text_list[doc_number]) if word in model.wv]
    word_vecs = {tuple(key): value for (key, value) in zip(document, words)}

    cls_alg = KMeans(n_clusters=50)
    #cls_alg = DBSCAN(eps=.000000001, metric='cosine', algorithm='brute', min_samples=1)
    #cls_alg = MeanShift()
    clusters = cls_alg.fit(doc_word_vectors[doc_number])

    vector_clusters = defaultdict(set)
    for i in range(len(clusters.labels_)):
        label = clusters.labels_[i]
        vector_clusters[label].add(tuple(doc_word_vectors[doc_number][i]))

    for cluster in vector_clusters.keys():
        print("Cluster {}:".format(cluster+1))
        for vector in vector_clusters[cluster]:
            print(word_vecs[vector])
        print("\n")


    return chooseCluster(vector_clusters)

def chooseCluster(vector_cluster):
    return vector_cluster[0]


# Main Function -------------------------------------------

def getVectors():
    print("Starting Script...")
    try:
        model = gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True) 
    except:
        print("Saved Model Information Not Found")
        return
    
    #text = xml2df("MathFeedsDataAll.xml")["Text"].as_matrix()
    clusters = []
    ratings = []
    files = ["MathFeedsDataBadExtract.xml", "MathFeedsDataAll.xml"]
    for x in range(1):
        text = xml2df(files[x])["Text"].as_matrix()
        docs = [np.array([model.wv[word] for word in clean_text_list(doc) if word in model.wv]) for doc in text if type(doc) == str]
        #clusters = get_clusters(3, docs, text, model)

        for x in range(text.length):
            clusters.append(get_clusters(x,docs,text,model))
            ratings.append(x)
    
    
    with open("DataSetX.txt", 'w') as f:
        f.write(json.dumps(clusters))

    with open("DataSetY.txt", 'w') as f:
        f.write(json.dumps(ratings))

#Run Above Functions
getVectors()