# rdt
RDT: Russian Distributional Thesaurus (Русский Дистрибутивный Тезаурус)


This package let you efficiently use word graph of the [Russian Distributional Thesaurus](http://nlpub.ru/rdt).

Quickstart
----

1. Download the pre-packed resource:
 ```
 wget http://panchenko.me/data/russe/rdt.pkl
 ```

2. Install dependencies, e.g.:
 ```
 pip install -r requirements.txt
 ```

3. Load the distributional thesaurus (specify path to the downloaded 'rdt.pkl' file):
 ```
 from rdt import RDT
 rdt = RDT(dt_pkl_fpath="rdt.pkl")
 ```
 Loading takes about 5 minutes and the resulting structure occupy around 1.3 Gb of RAM. This is however more efficient than parsing the CSV file into a dict in terms of both time and memory consumption. This implementation relies on marisa trie for storing keys and on numpy array for storing similarity scores. 

4. Search for nearest neighbours:
 ```
 for w,s in rdt.most_similar(u"граф"):
     print w,s
 ```
