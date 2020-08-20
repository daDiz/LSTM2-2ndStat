import matplotlib.pyplot as plt
import cPickle
import numpy as np

path0 = ['./rand-1/cst_1_0.0_128_0.01.pkl', './rand-2/cst_1_0.0_128_0.01.pkl',
'./earthquake/cst_1_0.0_64_1.0.pkl', './email/cst_1_0.0_256_0.01.pkl', './twitter/cst_1_0.0_256_0.01.pkl']
path1 = ['./rand-1/cst_1_0.1_128_0.01.pkl', './rand-2/cst_1_0.1_128_0.01.pkl',
'./earthquake/cst_1_0.1_64_1.0.pkl', './email/cst_1_100.0_256_0.01.pkl', './twitter/cst_1_1.0_256_0.01.pkl']
names = ['Rand-1', 'Rand-2', 'Earthquake', 'Email', 'Twitter']

for i in range(0,5):
    p0 = path0[i]
    p1 = path1[i]
    name = names[i]

    f0 = open(p0, 'rb')
    f1 = open(p1, 'rb')

    c0 = cPickle.load(f0)
    c1 = cPickle.load(f1)

    plt.plot(np.arange(1, len(c0)+1)*10, c0, alpha=1.0, label=name+': LC')
    plt.plot(np.arange(1, len(c1)+1)*10, c1, alpha=1.0, label=name+': LC+LK')


plt.xticks(np.arange(0,5000,1000),fontsize=14,weight='bold')
plt.yticks(np.arange(0,500,100),fontsize=14,weight='bold')
plt.legend(prop={'size':11, 'weight':'bold'}, ncol=2)

plt.xlabel('number of batches', fontsize=14, weight='bold')
plt.ylabel('cost', fontsize=14, weight='bold')

plt.show()
#plt.savefig('cst.pdf')
