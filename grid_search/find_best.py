beta = [0.1]#, 1, 10, 100, 1000]
dim = [64]#,128,256,512]
lr = [0.01]#, 0.1, 1.0, 10.0]

hit = [0,0,0]
para = [None, None, None]

for x in beta:
    for y in dim:
        for z in lr:
            f = open('hit_%s_%s_%s.txt' % (x,y,z))
            i = 0
            for line in f:
                h = float(line.strip('\n'))
                if h > hit[i]:
                    para[i] = ('beta:'+str(x),'dim:'+str(y),'lr:'+str(z))
                    hit[i] = h

                i += 1

print('hit @10, 20, 30')
print(para)
print(hit)
