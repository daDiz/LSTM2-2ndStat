from __future__ import print_function, division, absolute_import

import numpy as np
import cPickle
import matplotlib.pyplot as plt

def load_seq(file_name, format='pkl'):
    f = open(file_name)

    if format == 'pkl':
        x = cPickle.load(f)
    elif format == 'txt':
        x = np.array(map(int, f.readline().strip('\n').split()))

    return x


def calc_j(seq1, seq2, step):
    s1 = seq1[:step]
    s1_ = np.unique(s1)
    s2 = seq2[:step]
    s2_ = np.unique(s2)

    n1 = len(np.intersect1d(s1_, s2_))
    n2 = len(np.union1d(s1_, s2_))

    return float(n1) / float(n2)

file1 = './seq_real.pkl' # real
file2 = './seq_fake_2.pkl' # LC
file3 = './seq_fake_2nd.pkl' # LC+LK
file4 = './seq_fake_mark_mle_basis_exp.txt' # hawkes process
file5 = './seq_fake_rnnpp.pkl' # RNNPP

c1 = load_seq(file1)
c2 = load_seq(file2)
c3 = load_seq(file3)
c4 = load_seq(file4, format='txt')
c5 = load_seq(file5)

real = c1
fake = [c2, c3, c4, c5]
method = ['LC', 'LC+LK', 'Hawkes-Exp', 'RNNPP']

step = [100, 500, 1000, 2000]

for s in step:
    print('step: %s' % (s))
    for i in range(len(fake)):
        x = fake[i]
        m = method[i]
        print('%s js: %.3f' % (m, calc_j(c1, x, s)))
    print('--------------------------')


