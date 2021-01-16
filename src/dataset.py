# -*- coding: utf-8 -*-      
#--------------------------------------------
# Author:chenhao
# Date:2020-12-05 16:42
# Description:  
#--------------------------------------------
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, RandomSampler
from collections import defaultdict

class ml_1mTrainDataset(Dataset):
    def __init__(self, path, num_negs, seed):
        super(ml_1mTrainDataset, self).__init__()
        self.num_negs = num_negs
        random.seed(seed)
        np.random.seed(seed)

        self.users = []
        self.items = []
        self.labels = []

        user_sets_every = defaultdict(lambda : set())
        items_all = set()
        self.dataset = []

        #pos
        with open(path, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                if len(arr) >= 3:
                    user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])

                    user_sets_every[user].add(item)
                    items_all.add(item)

                    if rating > 0:
                        self.dataset.append((user, item, 1.0))
                line = f.readline()

        #negs
        # for user, item_sets in user_sets_every.items():
        #     neg_sets = items_all - item_sets
        #     try:
        #         for item in random.sample(neg_sets, self.num_negs*len(item_sets)): #every user add negtive = pos * num_negs
        #             self.dataset.append((user, item, 0.0))
        #     except:
        #         for item in neg_sets:
        #             self.dataset.append((user, item, 0.0))
        items_all = list(items_all)
        for user, item_sets in user_sets_every.items():

            for item in np.random.choice(items_all, self.num_negs*len(item_sets)):
                self.dataset.append((user, item, 0.0))




    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
        # return {'user': self.users[idx], 'item': self.items[idx], 'label': self.labels[idx]}

def ml_1mTrainDataLoader(path, num_negs, batch_size,  seed, worker_init, num_workers=1):
    train_data = ml_1mTrainDataset(path, num_negs, seed)
    train_sampler = RandomSampler(train_data)
    # return DataLoader(train_data, sampler=train_sampler, batch_size=batch_size,
    #                               num_workers=num_workers, worker_init_fn=worker_init,
    #                               pin_memory=True, shuffle=True)
    return DataLoader(train_data, batch_size=batch_size,
                                  num_workers=num_workers, worker_init_fn=worker_init,
                                  pin_memory=True, shuffle=True)

def ml_1mTestData(test_pos_path, test_neg_path):
    pos = []
    neg = []

    with open(test_pos_path, 'r') as fr:
        line = fr.readline()
        while line:
            row = line.strip().split('\t')
            if len(row) > 2:
                user, item = int(row[0]), int(row[1])
                pos.append([user, item])
            line = fr.readline()

    with open(test_neg_path, 'r') as fr:
        line = fr.readline()
        while line:
            row = line.strip().split('\t')
            if len(row) >= 2:
                neg.append([int(item) for item in row[1:]])
            line = fr.readline()

    assert len(pos) == len(neg)

    return pos, neg



