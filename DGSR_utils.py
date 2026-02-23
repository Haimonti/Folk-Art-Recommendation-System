#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/11/19 10:54
# @Author : ZM7
# @File : DGSR_utils
# @Software: PyCharm

import numpy as np
import sys

def eval_metric(all_top, random_rank=True):
    print('all_top',all_top)
    print('all_top',len(all_top))

    recall5, recall10, recall20, recall30, recall40, recall50, recall60, recall70, recall80, recall90, recall100, ndgg5, ndgg10, ndgg20, ndgg30, ndgg40, ndgg50, ndgg60, ndgg70, ndgg80,  ndgg90,  ndgg100 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [] 
    data_l = np.zeros((100, 7))
    for index in range(len(all_top)):
        print('index',index)
        prediction = (-all_top[index]).argsort(1).argsort(1)
        print('prediction',prediction)
        print('prediction',len(prediction))
        predictions = prediction[:, 0]
        print('predictions',predictions)

        print('predictions',len(predictions))

        for i, rank in enumerate(predictions):
            # print('rank',rank)
            if rank < 100:
                ndgg100.append(1 / np.log2(rank + 2))
                recall100.append(1)   
            else:
                ndgg100.append(0)
                recall100.append(0)                    
            if rank < 90:
                ndgg90.append(1 / np.log2(rank + 2))
                recall90.append(1) 
            else:
                ndgg90.append(0)
                recall90.append(0)                      
            if rank < 80:
                ndgg80.append(1 / np.log2(rank + 2))
                recall80.append(1)   
            else:
                ndgg80.append(0)
                recall80.append(0)                    
            if rank < 70:
                ndgg70.append(1 / np.log2(rank + 2))
                recall70.append(1)  
            else:
                ndgg70.append(0)
                recall70.append(0)                     
            if rank < 60:
                ndgg60.append(1 / np.log2(rank + 2))
                recall60.append(1)
            else:
                ndgg60.append(0)
                recall60.append(0)                   
            if rank < 50:
                ndgg50.append(1 / np.log2(rank + 2))
                recall50.append(1)
            else:
                ndgg50.append(0)
                recall50.append(0)        
            if rank < 40:
                ndgg40.append(1 / np.log2(rank + 2))
                recall40.append(1)
            else:
                ndgg40.append(0)
                recall40.append(0)    
            if rank < 30:
                ndgg30.append(1 / np.log2(rank + 2))
                recall30.append(1) 
            else:
                ndgg30.append(0)
                recall30.append(0)      
            if rank < 50:
                ndgg50.append(1 / np.log2(rank + 2))
                recall50.append(1)
            else:
                ndgg50.append(0)
                recall50.append(0)     
            if rank < 40:
                ndgg40.append(1 / np.log2(rank + 2))
                recall40.append(1)
            else:
                ndgg40.append(0)
                recall40.append(0)    
            if rank < 30:
                ndgg30.append(1 / np.log2(rank + 2))
                recall30.append(1)
            else:
                ndgg30.append(0)
                recall30.append(0)                                  
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
    return  np.mean(recall5), np.mean(recall10), np.mean(recall20), np.mean(recall30), np.mean(recall40), np.mean(recall50), np.mean(recall60), np.mean(recall70), np.mean(recall80), np.mean(recall90), np.mean(recall100), np.mean(ndgg5), np.mean(ndgg10), np.mean(ndgg20), np.mean(ndgg30), np.mean(ndgg40), np.mean(ndgg50), np.mean(ndgg60), np.mean(ndgg70), np.mean(ndgg80), np.mean(ndgg90), np.mean(ndgg100)   


def mkdir_if_not_exist(file_name):
    import os
    import shutil

    dir_name = os.path.dirname(file_name)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)


class Logger(object):
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass