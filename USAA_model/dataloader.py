import torch.utils.data as Data
import torch
import pickle,scipy
import numpy as np
import math
import gc
from prefetch_generator import BackgroundGenerator

class DataLoaderX(Data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

#filepath = 'D:\\学术相关\\007.CasCN-master\\dataset_weibo'
class MyDataset(Data.Dataset):
    def __init__(self, filepath):
        # 获得训练数据的总行
        x, A, y = pickle.load(open(filepath, 'rb'), encoding='utf-8')

        self.number = len(x)
        batch_A = []

        batch_time_interval_index_sample = []
        for i in range(len(A)):
            A_smp = A[i].todense().tolist()
            batch_A.append(A_smp)

        self.A = torch.tensor(batch_A,dtype=torch.float32)
        #x = list(map(int,x))
        #y = list(map(int, y))
        self.x = torch.tensor(x,dtype = torch.float32)
        self.y = torch.tensor(y,dtype=torch.long)

        gc.collect()

    def __len__(self):
        return self.number

    def __getitem__(self, idx):
        return self.x[idx],self.A[idx], self.y[idx]#,self.L[idx]
