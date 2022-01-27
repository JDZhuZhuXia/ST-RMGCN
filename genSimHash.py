import numpy as np
import torch
import pandas as pd
from collections import Counter
import math
import argparse
import time


parser = argparse.ArgumentParser()
parser.add_argument('--data',type=str,default='Data/metr-la.h5',help='data path')
parser.add_argument('--save',type=str,default='sim/metr-sim',help='save')
parser.add_argument('--n',type=str,default=8,help='number of neighbor')
parser.add_argument('--k',type=str,default=24,help='dimension of vector')

args = parser.parse_args()

def main():
    k = args.k
    n = args.n
    df = pd.read_hdf(args.data)
    data = np.array(df.values)
    data = data - np.median(data,axis=0)
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if(data[i,j]>=-0.5 and data[i,j]<=0.5):
                data[i,j] = 0
            elif(data[i,j]>0.5):
                data[i,j] = 1
            else:data[i,j] = -1
    #
    # data[data>0.5] = 1
    # data[data<-0.5] = -1
    # data[data<0.5 and data>-0.5] = 0
    data = data[:data.shape[0]-data.shape[0]%n,:]
    data = data.reshape(-1,n,data.shape[1])
    data = np.sum(data,axis=0)
    data[data>0] = 1
    data[data<0] = -1
    print(data.shape,data)
    
    Dis = np.zeros([data.shape[1],data.shape[1]])
    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            Dis[i,j] = n-Counter(data[:,i] - data[:,j])[0]

    np.save(args.save+'Dis.npy',Dis)

#     Dis = np.load('pems-simHashDis.npy')
    # 选择top个节点作为邻居
    top = n
    w_adj = np.zeros([Dis.shape[0],Dis.shape[0]])
    for i in range(Dis.shape[0]):
        a = Dis[i,:].argsort()[0:top]
        for j in range(top):
            w_adj[i, a[j]] = Dis[i,a[j]]

    for i in range(w_adj.shape[0]):
        adj_sum = sum(w_adj[i,:])

        for j in range(w_adj.shape[1]):
            if w_adj[i][j] != 0:
                w_adj[i][j] = w_adj[i][j]/adj_sum

    for i in range(w_adj.shape[0]):
        for j in range(w_adj.shape[1]):
            if w_adj[i][j] != 0:
                w_adj[i][j] = math.exp(-(w_adj[i][j]**2)/(0.12**2*2))
            w_adj[i][i] = 1
    print(w_adj[0])
    np.save(args.save+str(k)+'.npy',w_adj)

if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))