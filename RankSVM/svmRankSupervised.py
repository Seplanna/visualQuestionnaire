from svmRank import RankSVM
import os
from glob import glob
from utils import *
import subprocess


if __name__ == '__main__':
    data_ordering = GetOrderingFromImagePath("../DATA/ut-zap50k/ut-zap50k-data/image-path.mat")
    print("read ordering")
    #data_colour, data_gist = GetFeaturesGist("../DATA/ut-zap50k/ut-zap50k-data/ut-zap50k-lexi/zappos-feats-real-64x64.mat")
    labels1 = GetLabelsFromLexiFile("../DATA/ut-zap50k/ut-zap50k-data/ut-zap50k-lexi/zappos-labels-real-lexi.mat")
    print("read labels")
    dir  = "../DATA/ut-zap50k/ut-zap50k-images/"
    attr = 9
    labels = labels1[attr]
    dir_a = "../CycleGAN_shoes/supervised_test_AtoB" + str(attr + 1) + "/"
    dir_b = "../CycleGAN_shoes/supervised_test_BtoA" + str(attr + 1) + "/"
    os.makedirs(dir_a)
    os.makedirs(dir_b)
    for i in range(labels.shape[1]):
        print(i)
        data = labels[:,i]
        dir_to_write_1 = dir_a
        dir_to_write_2 = dir_b
        if (int(data[3]) > 1):
            dir_to_write_1 = dir_b
            dir_to_write_2 = dir_a
        subprocess.call(["cp", dir + data_ordering[int(data[0])], dir_to_write_1 + str(i) + ".jpg"])
        subprocess.call(["cp", dir + data_ordering[int(data[1])], dir_to_write_2 + str(i) + ".jpg"])




