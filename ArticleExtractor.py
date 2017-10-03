#Kaden Barlow
#Script for MathFeeds Research
#This script will take the XML data which includes the HTML of URLs already
#scraped and will extract the articles and then leave it in an XML file that
#can be loaded into a pandas DataFrame

#IMPORTS----------------------------------------------------------------------
import pandas as pd
import numpy as np
#For HTML extraction
import newspaper
#For parsing XML
import xml.etree.ElementTree as ET
from lxml import etree

#Functions I will use during the script --------------------------------------
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

def to_xml(df, filename=None):
    def row_to_xml(row):
        xml = '\n\t<Item>'
        for i, col_name in enumerate(row.index):
            #I am using CDATA here because there are lots of characters
            #that python isn't happy with
            #CData lets it be in the XML but won't check the characters
            xml += '\n\t\t<{0}><![CDATA[{1}]]></{0}>'.format(col_name, row.iloc[i])
        xml += '\n\t</Item>\n'
        return xml
    res = df.apply(row_to_xml, axis=1)

    if filename is None:
        return res
    with open(filename, 'w', encoding='utf-8') as f:
        #Root Tag is necessary for parsing
        f.write('<Data>')
        for row in res:
            f.write(row)
        f.write('</Data>')

pd.DataFrame.to_xml = to_xml



#Start of doing things -------------------------------------------------------
xml_data = input("File path of XML Data to extract?\n")
dataFrame = xml2df(xml_data)
#In the actual scraping I didn't pull the Domain but I use a regular expression here to put it in a separate column
dataFrame['Domain'] = dataFrame['URL'].str.extract('https*:\/\/(www\.)*(?P<Domain>(.)*(\.org|\.com|\.edu|\.net|\.gov))')['Domain']

with open(fname, "r") as fp:
    for line in fp:
        line = line.strip()
        line=line.decode('utf-8','ignore').encode("utf-8")

#So normally the Newspaper library requires you to download the article through their library and then parse it
#I have already downloaded all the information so I obviously didn't want to take the time and the processing to do that
#so I used the following work around, I did it in batches of 500 because this does take some time to finish
x = 0
y = input("Number of URL articles to extract?\n")
while x < y:
    #Add Status Updates Here :D
    if(x%100==0):
        print("Status: ", x/10, "%")
    article = newspaper.Article(url=dataFrame["URL"][x])
    article.set_html(dataFrame["HTML"][x])
    article.parse()
    dataFrame["HTML"][x] = article.text
    x= x+1

dataFrame.to_xml("Output.xml")
