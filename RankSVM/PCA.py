import numpy as np
import os
from sklearn.decomposition import PCA
import random
import subprocess
from utils import GetData, Gist
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--PCA_GIST', type = bool, default=False)
parser.add_argument('--PCA_AutoEncoder', type = bool, default=False)
parser.add_argument('--data_path', default="")
parser.add_argument('--n_features_embedding', type = int, default=-1)
parser.add_argument('--new_features_embedding', type = int, default=-1)
parser.add_argument('--result_dir', default="")

FLAGS, unparsed = parser.parse_known_args()

def AutoEncoderLatent(data_path, feature_dimension_embedding):
    data = GetData(data_path)
    random.shuffle(data)
    latent = np.zeros([len(data), feature_dimension_embedding])
    for d in range(len(data)):
        basename = os.path.basename(data[d]).split("_")[:-1]
        latent[d] = np.array([float(f) for f in basename])
    return data, latent

def GIST_latent(data_path, feature_dimension_embedding):
    data = GetData(data_path)
    random.shuffle(data)
    latent = np.zeros([len(data), feature_dimension_embedding])
    for d in range(len(data)):
        latent[d] = Gist(data[d])
    return data, latent

def PCA_(data, latent, feature_dimension, result_dir):
    pca = PCA(n_components=5)
    pca.fit(latent)
    #print(pca.singular_values_)
    latent1 = pca.fit_transform(latent)
    os.makedirs(result_dir)
    for i in range(feature_dimension):
        os.makedirs(result_dir + str(i) + "/")
    for d in range(len(data)):
        basename = os.path.basename(data[d]).split("_")[-1]
        for i in range(feature_dimension):
            subprocess.call(['cp', data[d], result_dir + str(i) + "/" +str(latent1[d][i]+10) +"_" + "_".join(str(f) for f in latent1[d]) + "_" + basename])

if FLAGS.PCA_GIST:
    data_path = FLAGS.data_path
    n_features_embedding = FLAGS.n_features_embedding
    n_features_new = FLAGS.new_features_embedding
    result_dir = FLAGS.result_dir
    data,latent = GIST_latent(data_path, n_features_embedding)
    PCA_(data, latent, n_features_new, result_dir)

if FLAGS.PCA_AutoEncoder:
    data_path = FLAGS.data_path
    n_features_embedding = FLAGS.n_features_embedding
    n_features_new = FLAGS.new_features_embedding
    result_dir = FLAGS.result_dir
    data,latent = AutoEncoderLatent(data_path, n_features_embedding)
    PCA_(data, latent, n_features_new, result_dir)



