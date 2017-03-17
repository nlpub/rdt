# -*- coding: utf-8 -*-

# change dt_pkl_fpath to path to the rdt.pkl file, which
# can be downloaded at http://panchenko.me/data/russe/rdt.pkl

from dt import RDT, DistributionalThesaurus

# load the distributional thesaurus
rdt = RDT(dt_pkl_fpath="rdt.pkl")
# test: retrieve nearest neighbours 
for w,s in rdt.most_similar(u"граф"):
    print w,s

# alternatively, you can use the code below and the file will be downloaded
# automatically at the first run

# load the distributional thesaurus
rdt = RDT()
# test: retrieve nearest neighbours 
for w,s in rdt.most_similar(u"граф"):
    print w,s

   
