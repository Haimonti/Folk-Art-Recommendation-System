#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/11/19 10:54
# @Author : ZM7
# @File : DGSR_utils
# @Software: PyCharm

import numpy as np
import sys
import math
'''
def eval_metric(all_top, random_rank=True):
    recall5, recall10, recall20, ndgg5, ndgg10, ndgg20 = [], [], [], [], [], []
    data_l = np.zeros((100, 7))
    for index in range(len(all_top)):
        prediction = (-all_top[index]).argsort(1).argsort(1)
        predictions = prediction[:, 0]
        for i, rank in enumerate(predictions):
            # data_l[per_length[i], 6] += 1
            if rank < 20:
                ndgg20.append(1 / np.log2(rank + 2))
                recall20.append(1)
            else:
                ndgg20.append(0)
                recall20.append(0)
            if rank < 10:
                ndgg10.append(1 / np.log2(rank + 2))
                recall10.append(1)
            else:
                ndgg10.append(0)
                recall10.append(0)
            if rank < 5:
                ndgg5.append(1 / np.log2(rank + 2))
                recall5.append(1)
            else:
                ndgg5.append(0)
                recall5.append(0)
    return np.mean(recall5), np.mean(recall10), np.mean(recall20), np.mean(ndgg5), np.mean(ndgg10), np.mean(ndgg20)
'''
def eval_metric(all_top):
    """
    Calculates Recall and NDCG for Top-K = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100].
    Assumes the positive target item is at index 0 of the score array.
    """
    ranks = []
    
    # all_top is a list of batches, each containing the scores of pos + neg items
    for batch_scores in all_top:
        for scores in batch_scores:
            # We sort descending by negating the scores. 
            # The target item is always at index 0. We find what rank index 0 landed in.
            rank = np.argsort(-scores).tolist().index(0)
            ranks.append(rank)
            
    k_list = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    recalls = []
    ndcgs = []
    
    total_samples = len(ranks)
    
    for k in k_list:
        # Recall @ k: 1 if rank < k else 0
        hits = sum(1 for r in ranks if r < k)
        recall = hits / total_samples if total_samples > 0 else 0.0
        
        # NDCG @ k: 1 / log2(rank + 2) if rank < k else 0
        ndcg = sum(1.0 / math.log2(r + 2) for r in ranks if r < k) / total_samples if total_samples > 0 else 0.0
        
        recalls.append(recall)
        ndcgs.append(ndcg)
        
    # Unpack the lists into exactly 22 return values
    return tuple(recalls + ndcgs)

def mkdir_if_not_exist(file_name):
    import os
    import shutil

    dir_name = os.path.dirname(file_name)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)


class Logger(object):
    """
    这个类的目的是尽可能不改变原始代码的情况下, 使得程序的输出同时打印在控制台和保存在文件中
    用法: 只需在程序中加入一行 `sys.stdout = Logger(log_file_path)` 即可
    """
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass