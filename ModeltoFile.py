#Imports-----------------------------------------------------------------------
import pandas as pd
import xml.etree.ElementTree as ET
from lxml import etree
import string
from collections import defaultdict
import gensim
import numpy as np


#Helper Functions--------------------------------------------------------------
def clean_word(w):
    strip_str = "()\".?!,;"
    new_word = "".join((c for c in w if c in string.printable))
    return new_word.strip(strip_str).lower()

def clean_text_list(doc):
    words = doc.split()
    clean_words = [clean_word(word) for word in words]
    return clean_words

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

def trainModel(fname):
    try:
        df = xml2df(fname)
    except:
        print("Data not found")
        return

    #Training the Model------------------------------------------------------------
    text = df['Text'].as_matrix()
    sentences = [clean_text_list(doc) for doc in text if type(doc) == str]
    model = gensim.models.Word2Vec(sentences, size=300, window=5, min_count=5)
    model.save('SavedModel')
    return

#End Script--------------------------------------------------------------------
