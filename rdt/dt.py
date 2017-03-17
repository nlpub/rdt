import marisa_trie
import numpy as np
from sklearn.externals import joblib

import codecs
from os.path import join, splitext, exists
from os import makedirs
import gzip
from heapq import heappop, heappush
from time import time
from traceback import format_exc
import cPickle as pickle
import urllib


SEP = "\t"
SEP_SCORE = ":"
SEP_LIST = ","
UNSEP = "_"
MIN_SIM = 0.0
KEYS = "keys.marisa"
SIMS = "sims"
VERBOSE = False
RDT_FPATH = "rdt.pkl"
RDT_URL = "http://panchenko.me/data/russe/rdt.pkl"

class RDT:
    """ Represents Russian Distributional Thesaurus (RDT). """
    
    def __init__(self, dt_pkl_fpath=RDT_FPATH):
        
        if not exists(dt_pkl_fpath):
            print "Downloading RDT: please wait, it can take several minutes..."
            rdt_file = urllib.URLopener()
            rdt_file.retrieve(RDT_URL, RDT_FPATH)
            if exists(RDT_FPATH): 
                print "Downloaded RDT file to:", RDT_FPATH
            else: 
                print "Download error: try again later or provide a valid RDT file."
                return
            
        self.dt = pickle.load(open(dt_pkl_fpath, "rb"))
        
        print "Testing the loaded model:"
        for w,s in self.dt.most_similar(u"граф", top_n=5): print w,s

    def most_similar(self, word, top_n=20):
        """ Get most similar terms of a word from the distributional thesaurus. """
        
        return self.dt.most_similar(word, top_n)
    
    def save(dt_fpath):
        pickle.dump(dt, open(dt_fpath, "wb"))

        
class DistributionalThesaurus:
    """ Represents a static distributional thesaurus, efficiently stored in memory. """
    
    def __init__(self, dt_dir):
        self.dt_dir = dt_dir
        
        self.keys_fpath = join(dt_dir, KEYS)
        self.sims_fpath = join(dt_dir, SIMS)

        if exists(self.keys_fpath) and exists(self.sims_fpath):
            print "Loading DT from:", dt_dir
            self.load_dt(dt_dir)
        else:
            print "Cannot load DT from:", dt_dir
            self.keys, self.sims = None, None
        
    def _iter_dt_word_word_score(self, dt_fpath):
        """ Iterates over a 'word_i<TAB>word_j<TAB>sim_ij' file,
        yielding (word_i, word_j, sim_ij) tuples. """

        with gzip.open(dt_fpath) if splitext(dt_fpath)[-1] == ".gz" else codecs.open(dt_fpath,"r","utf-8") as input_file:
            rel_num = 0
            for i, line in enumerate(input_file):
                try:
                    fields = line.split(SEP)
                    if len(fields) != 3: continue
                    sim = float(fields[2])
                    if sim < MIN_SIM: continue
                    rel_num += 1
                    word_i = fields[0].replace(SEP, UNSEP)
                    word_j = fields[1].replace(SEP, UNSEP)
                    yield (word_i, word_j, sim)
                except:
                    print format_exc()
                    if VERBOSE: print "bad line:", i, line

            print "# relations loaded:", rel_num, "out of", i + 1

    def _iter_dt_word_neighbors(self, dt_fpath):
        """ Iterates over a 'word_i<TAB>word_i:sim_ij,word_k:sim_ik'
        file, yielding (word_i, word_j, sim_ij) tuples. """

        with gzip.open(dt_fpath) if splitext(dt_fpath)[-1] == ".gz" else codecs.open(dt_fpath,"r","utf-8") as input_file:
            rel_num = 0
            for i, line in enumerate(input_file):
                try:
                    word_i, neighbors = line.split(SEP)
                    word_i = word_i.replace(SEP, UNSEP)
                    for word_j_sim_ij in neighbors.split(SEP_LIST):
                        word_j, sim_ij = word_j_sim_ij.split(SEP_SCORE) 
                        word_j = word_j.replace(SEP, UNSEP)
                        sim_ij = float(sim_ij)
                        if sim_ij < MIN_SIM: continue
                        rel_num += 1
                        yield word_i, word_j, sim_ij
                except:
                    print format_exc()
                    if VERBOSE: print "bad line:", i, line

            print "# relations loaded:", rel_num
            print "# lines processed:", i + 1
            
    def build_dt(self, dt_fpath, dt_format="word_word_score"):
        """ Builds and persists data structures for storage of DT:
        a marisa trie for keys + a numpy array for scores. 
        The 'dt_format' is in 'word_word_score', 'word_neighbors'. 
        The first format corresponds to a CSV file 'word_i<TAB>word_j<TAB>sim_ij'.
        The second format corresponds to a CSV file 'word_i<TAB>word_j:sim_ij,work_k:sim_ik,...'
        """

        tic=time()
        if dt_format == "word_word_score":
            iter_dt = self._iter_dt_word_word_score
        elif dt_format == "word_neighbors":
            iter_dt = self._iter_dt_word_neighbors
        else: 
            iter_dt = self._iter_dt_word_word_score
            
        if not exists(self.dt_dir): makedirs(self.dt_dir)
        self.keys = marisa_trie.Trie([w1 + SEP + w2 for w1, w2, _ in iter_dt(dt_fpath)])
        self.keys.save(self.keys_fpath)
        print "DT keys:", self.keys_fpath

        self.sims = np.zeros(len(self.keys), dtype='Float16')
        for i, (w1, w2, sim) in enumerate(iter_dt(dt_fpath)):
            self.sims[self.keys.key_id(w1 + SEP + w2)] = sim
        joblib.dump(self.sims, self.sims_fpath)
        print "DT scores:", self.sims_fpath
        print "Building DT took", time()-tic, "sec."
        
    def load_dt(self, input_dir):
        """ Loads a pre-built distributional thesaurus structure. """

        tic = time()
        self.keys = marisa_trie.Trie()
        self.keys.load(self.keys_fpath)
        print "Loaded %d keys: %s" % (len(self.keys.items()), self.keys_fpath)
        self.sims = joblib.load(self.sims_fpath)
        print "Loaded %d scores: %s" % (self.sims.size, self.sims_fpath)
        print "Loading DT took", time()-tic, "sec."

    def most_similar(self, word, top_n=20):
        """ Get most similar terms of a word from the distributional thesaurus. """
        
        if self.keys is None or self.sims is None:
            print "Model is not loaded: load or build a DT first."
            return []
        # using heappush/heappop to sort by score
        h = []
        for pair in self.keys.keys(word + SEP):
            heappush(h, (-self.sims[self.keys[pair]], pair) )

        res = [ heappop(h) for i in range(len(h))]
        res = [(pair.split(SEP)[-1], -sim) for sim, pair in res]
        return res[:top_n]
    
