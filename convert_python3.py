#!/usr/bin/env python3

from dt import DistributionalThesaurus
import pickle
import sys

with open(sys.argv[1], 'rb') as f:
    dt = pickle.load(f, encoding='bytes')

fields = vars(dt)

dt.keys_fpath = fields[b'keys_fpath']
dt.sims_fpath = fields[b'sims_fpath']
dt.dt_dir = fields[b'dt_dir']
dt.keys = fields[b'keys']
dt.sims = fields[b'sims']

with open(sys.argv[1] + '.3', 'wb') as f:
    pickle.dump(dt, f)
