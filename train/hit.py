from __future__ import print_function, absolute_import, division
import cPickle
import sys
import numpy as np
from collections import Counter


class Hit():
    def __init__(self, data_file, seq_file):
        self._data_file = data_file
        self._seq_file = seq_file
        self._num_events = 0
        self._rate = 0
        self._data = None
        self._seq = None
        self._score = None

    @property
    def seq(self):
        return self._seq

    @property
    def data(self):
        return self._data

    @property
    def rate(self):
        return self._rate

    @property
    def score(self):
        return self._score

    def load(self):
        with open(self._data_file, 'r') as file:
            self._data = cPickle.load(file)

        with open(self._seq_file, 'r') as file:
            seq = cPickle.load(file)

        self._num_events = len(self._data)
        k = len(seq) - self._num_events
        self._seq = seq[k:]

    def hit_at(self, top=5):
        #h = lambda x: Counter(x).most_common(top)
        #h = lambda x: list(zip(*sorted(enumerate(x), key=lambda x: x[1]))[0])[-top:]
        h = lambda x: np.array(x).argsort()[::-1][:top]
        pred = map(h, self._data)
        #for i in range(len(self._data)):
        #    print(pred[i])
        #    print(self._seq[i]-1)
        #print(sorted(self._data[0])[::-1][:top])
        #print(self._data[0][pred[0][0]])
        score = [1.0 if self._seq[i]-1 in pred[i] else 0.0 for i in range(self._num_events)]
        self._score = score
        self._rate = sum(self._score)/self._num_events

    def hit_at_sample(self, top=5):
        h = lambda x: map(lambda y: y[0], Counter(x).most_common(top))
        #h = lambda x: list(zip(*sorted(enumerate(x), key=lambda x: x[1]))[0])[-top:]
        #h = lambda x: np.array(x).argsort()[::-1][:top]
        pred = map(h, self._data)
        #print(pred)
        #for i in range(len(self._data)):
        #    print(pred[i])
        #    print(self._seq[i]-1)
        #print(sorted(self._data[0])[::-1][:top])
        #print(self._data[0][pred[0][0]])
        score = [1.0 if self._seq[i]-1 in pred[i] else 0.0 for i in range(self._num_events)]
        self._score = score
        self._rate = sum(self._score)/self._num_events


    def hit_at_pct(self, pct=0.1):
        N = len(self._data[0])
        top = int(N * pct)
        h = lambda x: list(zip(*sorted(enumerate(x), key=lambda x: x[1]))[0])[-top:]
        pred = map(h, self._data)
        score = [1.0 if self._seq[i]-1 in pred[i] else 0.0 for i in range(self._num_events)]
        self._score = score
        self._rate = sum(self._score)/self._num_events



if __name__=='__main__':
    hr = Hit('./dist_fake.pkl', './seq_real.pkl')
    hr.load()

    for h in [10,20,30]:
        hr.hit_at(h)
        print(hr.rate)

