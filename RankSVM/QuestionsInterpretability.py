from Evaluation import DevideFeatureIntoBinsFromData, RecieveBinOfTheFeature
from utils import GetData, get_image, imsave
from Embedding import *
import numpy as np
import os
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--CreateReal1', type = bool, default=False)
parser.add_argument('--data_path', default='')
parser.add_argument('--result_path', default='')
parser.add_argument('--n_features', type=int, default=-1)
parser.add_argument('--n_bins', type=int, default=-1)
parser.add_argument('--n_pictures_per_bin', type=int, default=-1)
parser.add_argument('--embedding', default='')

parser.add_argument('--CheckTheQuestionnaire', type = bool, default=False)
parser.add_argument('--data_label_path', default='')

FLAGS, unparsed = parser.parse_known_args()

def ReturnImagesCLosedToMedian(features_in_bins, data_by_bins, data, n_bins, n_pictures_per_bin):
    closest_to_median_images = [[] for i in range(n_bins)]
    features_in_bins = np.array(features_in_bins)
    median = np.median(features_in_bins, axis=0)
    for bin in range(n_bins):
        feature_in_bin = features_in_bins[data_by_bins[bin]]
        feature_in_bin -= median
        norms = np.linalg.norm(feature_in_bin, axis=1)
        argsort = np.argsort(norms)
        n_pictures_per_bin_ = n_pictures_per_bin
        if (n_pictures_per_bin < 0):
            n_pictures_per_bin_ = len(data_by_bins[bin])
        for i in range(n_pictures_per_bin_):
            closest_to_median_images[bin].append(data[data_by_bins[bin][argsort[i]]])
    return closest_to_median_images

def Real1_one_feature_cluster(data_path, feature_n, n_pictures_per_bin, Embedding):
    n_clusters=2
    data_ = [GetData(data_path + str(n_clusters * (feature_n + 1) + i)) for i in range(n_clusters)]

    data_by_bins = []
    len_ = 0
    for i in range(n_clusters):
        data_by_bins.append([j + len_ for j in range(len(data_[i]))])
        len_ += len(data_[i])
    features_in_bins = []
    data = []
    for i in range(n_clusters):
        data[i] += data_[i]
    for d in range(len(data)):
        features = Embedding(data[d])
        features = [features[f] for f in range(features.shape[0]) if f != feature_n]
        features_in_bins.append(features)
    return ReturnImagesCLosedToMedian(features_in_bins, data_by_bins, data, n_clusters, n_pictures_per_bin)

def  Real1_one_feature_from_given_features(features, data, n_bins, n_pictures_per_bin, feature_n):
    features_in_bins = []
    feature_values = []
    for d in range(len(data)):
        features_= features[d]
        feature_value = features_[feature_n]
        feature_values.append(feature_value)
        features_ = [features_[f] for f in range(features_.shape[0]) if f != feature_n]
        features_in_bins.append(features_)


    data_by_bins = [[] for i in range(n_bins)]
    bins = DevideFeatureIntoBinsFromData(feature_values, n_bins)
    for d in range(len(feature_values)):
        bin = RecieveBinOfTheFeature(feature_values[d], bins, n_bins)
        data_by_bins[bin].append(d)
    return ReturnImagesCLosedToMedian(features_in_bins, data_by_bins, data, n_bins, n_pictures_per_bin), data_by_bins

def Real1_one_feature(data_path, feature_n, n_bins, n_pictures_per_bin, Embedding):

    #bins = DevideFeatureIntoBins(data_path, n_bins, 0)
    data = GetData(data_path)
    features_in_bins = []
    for d in range(len(data)):
        features = Embedding(data[d])
        features_in_bins.append(features)
    return Real1_one_feature_from_given_features(features_in_bins, data, n_bins, n_pictures_per_bin, feature_n)


def Real1(data_path, n_features, n_bins, n_pictures_per_bin, Embedding, result_dir):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for f in range(n_features):
        images_to_save = []
        images, _ = Real1_one_feature(data_path + str(f) + "/", f, n_bins, n_pictures_per_bin, Embedding)
        for b in range(n_bins):
            os.makedirs(result_dir + str(f) + "/" + str(b) + "/")
            for im in images[b]:
                subprocess.call(['cp', im, result_dir + str(f) + "/" + str(b) + "/" + os.path.basename(im)])
                images_to_save += [get_image(im, 110, 150,
                                    resize_height=64, resize_width=64)]
        imsave(np.array(images_to_save), [n_bins, n_pictures_per_bin], result_dir + str(f) + ".jpg")
#----------------------------------------------------------------------------------------------------------------------
def SimulateUser(data_path, data_label_path, Embedding, n_features):
    data_label_ = GetData(data_label_path)
    data_label = {}
    for d in data_label_:
        name = os.path.basename(d).split("_")[-1]
        data_label[name] = Embedding(d)
    data = GetData(data_path)
    res = np.zeros(n_features)
    for d in data:
        res += data_label[os.path.basename(d).split("_")[-1]]
    return res

if FLAGS.CheckTheQuestionnaire:
    data_path = FLAGS.data_path
    data_label_path = FLAGS.data_label_path
    n_features = FLAGS.n_features
    n_bins = FLAGS.n_bins
    embedding_type = FLAGS.embedding
    #result = []
    for question in range(n_features):
        r = []
        for b in range(n_bins):
            r.append(SimulateUser(data_path + str(question) + "/" + str(b) + "/", data_label_path, Embedding(embedding_type),
                                  n_features))
        print("QUESTION = ", question)
        print(r)




if FLAGS.CreateReal1:
   data_path = FLAGS.data_path
   n_features = FLAGS.n_features
   n_bins = FLAGS.n_bins
   n_pictures_per_bin = FLAGS.n_pictures_per_bin
   embedding_type = FLAGS.embedding
   result_dir = FLAGS.result_path

   Real1(data_path, n_features, n_bins, n_pictures_per_bin, Embedding(embedding_type), result_dir)





