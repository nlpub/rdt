# Loading the RDT into a hash table:
# Noe that this is less memory and cpu efficient 
# as compared to using rdt.pkl (based on marisa_trie and numpy)

import codecs
from collections import defaultdict
from traceback import format_exc
from time import time 
import gzip 
from os.path import splitext

# enter path to graph of words here
dt_fpath = "all.norm-sz500-w10-cb0-it3-min5.w2v.vocab_1100000_similar250.gz"
VERBOSE = False
SEP = "\t"
SEP_SCORE = ":"
SEP_LIST = ","
UNSEP = "_"
MIN_SIM = 0.0

tic = time()

with gzip.open(dt_fpath) if splitext(dt_fpath)[-1] == ".gz" else codecs.open(dt_fpath,"r","utf-8") as input_file:
    dt = defaultdict(lambda: defaultdict(float))
    rel_num = 0
    for i, line in enumerate(input_file):
        #if i > 10: break
        try:
            word_i, neighbors = line.split(SEP)
            word_i = word_i.replace(SEP, UNSEP)
            for word_j_sim_ij in neighbors.split(SEP_LIST):
                word_j, sim_ij = word_j_sim_ij.split(SEP_SCORE) 
                word_j = word_j.replace(SEP, UNSEP)
                sim_ij = float(sim_ij)
                if sim_ij < MIN_SIM: continue
                rel_num += 1
                dt[word_i][word_j] = sim_ij
        except:
            print(format_exc())
            if VERBOSE: print("bad line:", i, line)

print(time()-tic, "sec.")

print("Sample entries:")
i = 0
for w1 in dt:
    for w2 in dt[w1]:
        print(w1, w2, dt[w1][w2])
        i += 1
    if i > 1000: break
