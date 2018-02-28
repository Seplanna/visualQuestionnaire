import numpy as np
import os
from sklearn.decomposition import PCA
import random
import subprocess
from utils import GetData, Gist
from Embedding import *
import argparse

from Evaluation import DevideFeatureIntoBins

parser = argparse.ArgumentParser()

parser.add_argument('--PCA_GIST', type = bool, default=False)
parser.add_argument('--PCA_AutoEncoder', type = bool, default=False)
parser.add_argument('--data_path', default="")
parser.add_argument('--n_features_embedding', type = int, default=-1)
parser.add_argument('--new_features_embedding', type = int, default=-1)
parser.add_argument('--result_dir', default="")

parser.add_argument('--My', type = bool, default=False)
parser.add_argument('--svm', default="")

parser.add_argument('--AutoEncoderByDirection', type = bool, default=False)
parser.add_argument('--direction_path', default='')
parser.add_argument('--embedding', default='')



parser.add_argument('--check_interpretability', type = bool, default=False)
parser.add_argument('--n_bins', type = int, default=-1)
parser.add_argument('--step', type = int, default=-1)



FLAGS, unparsed = parser.parse_known_args()

def My_latent(data_path, svm_path, feature_dimension_embedding, result_dir):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    data = GetData(data_path)
    svm = []
    for i in range(feature_dimension_embedding):
        svm.append(np.genfromtxt(svm_path + str(i + 1) + ".txt"))

    for i in range(feature_dimension_embedding):
        os.makedirs(result_dir + str(i) + "/")
    for d in range(len(data)):
        g = Gist(data[d])
        latent = np.array([np.dot(svm[i], g) for i in range(feature_dimension_embedding)])
        basename = os.path.basename(data[d]).split("_")[-1]
        for i in range(feature_dimension_embedding):
            subprocess.call(['cp', data[d], result_dir + str(i) + "/" +str(latent[i]+10) +"_" + "_".join(str(f) for f in latent) + "_" + basename])
    return data

def AutoEncoderRankingByDirection(data, latent_, direction_path, feature_dimension_embedding, result_dir):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    direction = []
    for i in range(feature_dimension_embedding):
        direction.append(np.genfromtxt(direction_path + str(2*(i+1)) + "_" + str(2*(i+1)+1) + ".txt"))

    for i in range(feature_dimension_embedding):
        os.makedirs(result_dir + str(i) + "/")
    for d in range(len(data)):
        latent = np.array([np.dot(direction[i], latent_[d]) for i in range(feature_dimension_embedding)])
        basename = os.path.basename(data[d]).split("_")[-1]
        for i in range(feature_dimension_embedding):
            subprocess.call(['cp', data[d], result_dir + str(i) + "/" +str(latent[i]+10) +"_" + "_".join(str(f) for f in latent) + "_" + basename])
    return data


def AutoEncoderLatent(data_path, feature_dimension_embedding, Embedding):
    data = GetData(data_path)
    random.shuffle(data)
    latent = np.zeros([len(data), feature_dimension_embedding])
    for d in range(len(data)):
        latent[d] = Embedding(data[d])
    return data, latent

def GIST_latent(data_path, feature_dimension_embedding):
    data = GetData(data_path)
    random.shuffle(data)
    latent = np.zeros([len(data), feature_dimension_embedding])
    for d in range(len(data)):
        latent[d] = Gist(data[d])
    return data, latent

def PCA_(data, latent, feature_dimension,result_dir):
    latent_ = latent.T
    normed = (latent_ - latent_.mean(axis=0))/ (latent_.std(axis=0)+1e-5)
    normed = normed.T

    pca = PCA(n_components=feature_dimension)
    pca.fit(normed)
    #print(pca.singular_values_)
    latent1 = pca.fit_transform(normed)
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

    data,latent = AutoEncoderLatent(data_path, n_features_embedding, Embedding(FLAGS.embedding))
    PCA_(data, latent, n_features_new, result_dir)

if FLAGS.My:
    data_path = FLAGS.data_path
    n_features_embedding = FLAGS.n_features_embedding
    result_dir = FLAGS.result_dir
    svm_path = FLAGS.svm
    data = My_latent(data_path, svm_path, n_features_embedding, result_dir)

if FLAGS.AutoEncoderByDirection:
    data_path = FLAGS.data_path
    n_features_embedding = FLAGS.n_features_embedding
    result_dir = FLAGS.result_dir
    direction_path = FLAGS.direction_path
    data, latent = AutoEncoderLatent(data_path, 10, Embedding(FLAGS.embedding))
    print(direction_path)
    AutoEncoderRankingByDirection(data, latent, direction_path, n_features_embedding, result_dir)

if FLAGS.check_interpretability:
    data_path = FLAGS.data_path
    n_bins = FLAGS.n_bins
    result_dir = FLAGS.result_dir
    step = FLAGS.step

    dir1 = result_dir + str(2*step) + "/"
    dir2 = result_dir + str(2 * step + 1) + "/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    for i in range(n_bins):
        os.makedirs(result_dir + str(step) + "_" + str(i))

    bins = DevideFeatureIntoBins(data_path, n_bins, 0)
    data=GetData(data_path)
    for d in data:
        basename = os.path.basename(d)
        basename = basename.split("_")
        #print(bins, basename[0], d)
        bin = 0
        while(bin < n_bins and (float(basename[0]) > bins[bin])):
            bin += 1
        subprocess.call(['cp', d, result_dir + str(step) + "_" + str(bin) + "/" + "_".join(basename[1:])])





