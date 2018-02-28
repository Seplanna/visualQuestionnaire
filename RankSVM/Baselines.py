import numpy as np
import os
import argparse

from utils import GetData
from Embedding import *

parser = argparse.ArgumentParser()
parser.add_argument('--recieve_vector_of_difference_between_clusters', type = bool, default=False)
parser.add_argument('--data_path', default='')
parser.add_argument('--result_path', default='')
parser.add_argument('--n_features', type=int, default=-1)
parser.add_argument('--embedding', default='')


FLAGS, unparsed = parser.parse_known_args()
#-------------------------------------Without RankSVM--------------------------------
def Get_vector_of_difference_between_clusters(data_path, result_path, step, n_features, Embedding):
    #data_path=FLAGS.data_path
    #result_path=FLAGS.result_dir
    #step=FLAGS.step
    #n_features = FLAGS.n_features_embedding

    data1 = GetData(data_path + str(2*step) + "/")
    print(data_path + str(2*step) + "/")
    cluster1_mean = np.zeros(n_features)
    for d in data1:
        cluster1_mean += Embedding(d)
    cluster1_mean /= len(data1)

    data2 = GetData(data_path + str(2*step + 1) + "/")
    cluster2_mean = np.zeros(n_features)
    for d in data2:
        cluster2_mean += Embedding(d)
    cluster2_mean /= len(data2)

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    np.savetxt(result_path + str(2*step) + "_" + str(2*step+1) + ".txt", cluster2_mean - cluster1_mean)
    np.savetxt(result_path + str(2*step) +".txt", cluster1_mean)
    np.savetxt(result_path + str(2 * step + 1) + ".txt", cluster2_mean)

if FLAGS.recieve_vector_of_difference_between_clusters:
    n_features = FLAGS.n_features
    for f in range(n_features):
        Get_vector_of_difference_between_clusters(FLAGS.data_path, FLAGS.result_path, f+1, 10, Embedding(FLAGS.embedding))

