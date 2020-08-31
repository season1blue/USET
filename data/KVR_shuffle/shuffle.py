import torch

import os
import json
import random
import argparse
import numpy as np

if __name__=="__main__":
    # path = 'train.json'
    ftrain = open('../KVR/train.json')
    fdev = open("../KVR/dev.json")
    ftest = open("../KVR/test.json")

    # with open(path)as ft:
    train_data = json.load(ftrain)
    dev_data = json.load(fdev)
    test_data = json.load(ftest)

    all_data = train_data+ dev_data+ test_data
    np.random.shuffle(all_data)

    dump_train = open("train.json","w")
    dump_test = open("test.json","w")
    dump_dev = open("dev.json","w")

    # for index in range(len(all_data)):
    #     if(index<len(all_data)*0.1):
    #         json.dump(all_data[index],dump_test)
    #     elif index<len(all_data)*0.2 :
    #         json.dump(all_data[index],dump_dev)
    #     else:
    #         json.dump(all_data[index],dump_train)

    json.dump(all_data[0:len(all_data)*0.1],dump_test)

    # test_end_index = len(all_data)*0.1
    # shuffle_test = all_data[0:test_end_index]
    # dev_end_index = test_end_index + len(all_data)*0.1
    # shuffle_dev = all_data[test_end_index:dev_end_index]
    # shuffle_train = all_data[dev_end_index:]

    # json.dump(shuffle_train,ftrain)
    # json.dump(shuffle_dev,fdev)
    # json.dump(shuffle_test,ftest)

    # print(len(all_data))
    # print(len(train_data))
    # print(len(dev_data))
    # print(len(test_data))
