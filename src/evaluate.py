# -*- coding: utf-8 -*-      
#--------------------------------------------
# Author:chenhao
# Date:2020-12-06 18:02
# Description:  
#--------------------------------------------
import torch
import heapq
import math
import numpy as np

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0

def evaluate_one_data(model, data, device, topk):
    user, pos, negs = data
    items = [pos] + negs
    users = [user] * len(items)
    users = torch.Tensor(users).to(device)
    items = torch.Tensor(items).to(device)
    with torch.no_grad():
        scors = model(users, items)
        # print(scors)
    item_score = dict(zip(items, scors))
    ranklist = heapq.nlargest(topk, item_score, key=item_score.get)
    hr = getHitRatio(ranklist, pos)
    ndcg = getNDCG(ranklist, pos)
    return hr, ndcg

def evaluate(model, test_data, device, topk=3):
    hits = []
    ndcgs = []
    for data in test_data:
        hr, ndcg = evaluate_one_data(model, data, device, topk)
        hits.append(hr)
        ndcgs.append(ndcg)

    hit_res = np.array(hits).mean()
    ndcg_res = np.array(ndcgs).mean()

    return hit_res, ndcg_res
