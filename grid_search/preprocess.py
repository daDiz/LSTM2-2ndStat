from __future__ import division, print_function, absolute_import
import numpy as np
import cPickle
import math
import networkx as nx
import argparse

#########################################
parser = argparse.ArgumentParser(description='preprocess')

## optional
parser.add_argument('name', type=str, help='dataset name, e.g. earthquake, email, twitter, rand-1, rand-2')

args = parser.parse_args()

data_name = args.name

######## preprocess ################
def save_time(file_name, seq):
    with open(file_name, 'w') as file:
        for x in seq:
            for i in range(len(x)):
                if i == len(x)-1:
                    file.write('%.6f\n' % x[i])
                else:
                    file.write('%.6f,' % x[i])

def save_mark(file_name, seq):
    with open(file_name, 'w') as file:
        for x in seq:
            for i in range(len(x)):
                if i == len(x)-1:
                    file.write('%d\n' % x[i])
                else:
                    file.write('%d,' % x[i])

######## preprocess timeseries ###########
def load_data(mark_file_name):
    mark = []
    with open(mark_file_name) as file:
        for line in file:
            tmp = map(lambda x: int(x)-1, line.strip('\n').split(','))
            mark.append(tmp)

    return mark

# count number of events at each node in each time period (e.g. 30 days)
def count_event(mark, N):
    m = len(mark)
    bucket = np.zeros((m, N))
    for i in range(m):
        for j in range(N):
            cur = np.array(mark[i])
            bucket[i,j] = np.sum(cur==j)

    return bucket

def build_timeseries(bucket, window_size):
    m = len(bucket)
    N = len(bucket[0])
    timeseries = []
    for i in range(m):
        st = max(i-window_size, 0)
        tmp = []
        n_padding = -min(i-window_size, 0)

        #pad zeros at the front if necessary
        for j in range(0, n_padding):
            tmp.append(list(np.zeros(N)))
        for j in range(st, i):
            tmp.append(list(bucket[j]))

        timeseries.append(tmp)

    return np.array(timeseries)

# pair-wise shortest distance
# given a graph, write the pair-wise shortest distance (N-by-N array) to a file
def shortest_d(graph_name, out_name, normalize=True):
    g = nx.read_edgelist(graph_name, nodetype=int)
    length = dict(nx.all_pairs_shortest_path_length(g))
    dia = float(nx.diameter(g))
    arr = []
    keys = g.nodes
    for i in keys:
        tmp = []
        for j in keys:
            if normalize:
                tmp.append(length[i][j]/dia)
            else:
                tmp.append(length[i][j])
        arr.append(tmp)

    arr = np.array(arr)
    with open(out_name, 'w') as file:
        cPickle.dump(arr, file)

######## preprocess final ##############


####################
# main
###################

###### preprocess ###########
if data_name == 'earthquake':
    sub_name = 'eq_1997-2018'
    time_step = 3600.0 * 24 * 30 # time window size (in seconds)
    N = 648 # num of nodes in the network
elif data_name == 'email':
    sub_name = 'enron_0'
    time_step = 3600.0 * 24 * 1
    N = 2634
elif data_name == 'twitter':
    sub_name = 'safrica'
    time_step = 3600.0 * 24 * 1
    N = 393
elif data_name == 'rand-1':
    sub_name = 'rand_0'
    time_step = 0.01
    N = 291
elif data_name == 'rand-2':
    sub_name = 'rand_0'
    time_step = 0.01
    N = 1599
else:
    raise Exception('Unknown dataset. Please provide your own dataset.')


with open('../datasets/%s/seq_%s.txt' % (data_name, sub_name)) as file:
    mark = np.array(map(int, file.readline().strip('\n').split()))

with open('../datasets/%s/%s_time.txt' % (data_name, sub_name)) as file:
    time = np.array(map(float, file.readline().strip('\n').split()))

mark_ = []
time_ = []
start = time[0]
i = 0

while i < len(time):
    tmp_t = []
    tmp_m = []

    #print('start %f' % start)
    while i < len(time) and time[i]-start <= time_step:
        #print(mark[i])
        tmp_t.append(time[i]-start)
        tmp_m.append(mark[i])
        i += 1
    time_.append(tmp_t)
    mark_.append(tmp_m)
    if i < len(time):
        start = time[i]

time = time_
mark = mark_

tmax = np.max([v for a in time for v in a])
print('Tmax is %.6f' % tmax)
print('Timestep is %f' % time_step)

save_mark('./data/mark_%s.csv' % (sub_name), mark)
save_time('./data/time_%s.csv' % (sub_name), time)


######### preprocess timeseries ##########
window_size = 30 # num of windows for lstm 1, i.e. long range dependency
mark = load_data('./data/mark_%s.csv' % (sub_name))
bucket = count_event(mark, N)
timeseries = build_timeseries(bucket, window_size)
#print(timeseries.shape)
with open('./data/timeseries_%s.pkl' % (sub_name), 'w') as file:
    cPickle.dump(timeseries, file)



######### preprocess final #############
mark = load_data('./data/mark_%s.csv' % (sub_name))

with open('../datasets/%s/seq_%s.txt' % (data_name, sub_name)) as file:
    mark_global = np.array(map(int, file.readline().strip('\n').split()))-1


#N = 648 # num of nodes in the network
time_step = 32 + 1 # num of events considered in lstm 2

train_test_split = True
split = 0.7

n_global = len(mark_global)

n = len(mark)
idx = 0
event = []
background = []
for i in range(n):
    mark_i = mark[i]
    m = len(mark_i)
    #print(i)
    for j in range(m):
        if idx + time_step > n_global:
            break

        event.append(mark_global[idx:idx+time_step])
        background.append(i)

        idx += 1

event = np.array(event)
background = np.array(background)

if train_test_split:
    print('train test split ...')
    nn = int(math.ceil(len(event)*split))
    event_train = event[:nn]
    event_test = event[nn:]
    background_train = background[:nn]
    background_test = background[nn:]
    with open('./data/event_%s_train.pkl' % (sub_name), 'w') as file:
        cPickle.dump(event_train, file)
    with open('./data/event_%s_test.pkl' % (sub_name), 'w') as file:
        cPickle.dump(event_test, file)
    with open('./data/background_%s_train.pkl' % (sub_name), 'w') as file:
        cPickle.dump(background_train, file)
    with open('./data/background_%s_test.pkl' % (sub_name), 'w') as file:
        cPickle.dump(background_test, file)
else:
    with open('./data/event_%s.pkl' % (sub_name), 'w') as file:
        cPickle.dump(event, file)

    with open('./data/background_%s.pkl' % (sub_name), 'w') as file:
        cPickle.dump(background, file)


############ build shortest distance matrix #########
print('constructing shortest distance matrix ...')
shortest_d('../datasets/%s/%s.txt' % (data_name, sub_name), './data/shortest.pkl', normalize=False)


