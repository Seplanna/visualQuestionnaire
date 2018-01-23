from glob import glob
import os
import numpy as np
import random

def GetFeatures(data_path, n_features):
    data = glob(os.path.join(data_path, "*.jpg"))
    res = {}
    for d in data:
        basename = os.path.basename(d)
        features = basename.split("_")[-n_features - 1:-1]
        features = np.array([float(f) for f in features])
        res[basename.split("_")[-1]] = features
    return res


def GetImagesInBins(anchorImage_n, n_features, n_bins):
    data=[[] for f in range(n_features)]
    d_ = "../RankSVM_DATA/Shoes/" + str(n_bins) + "/Histogramm/CycleGan/DATA/"

    for feature in range(n_features):
        for b in range(n_bins):
            file = d_ + str(anchorImage_n) + "_" + str(feature) + "_" + str(b)+"/"
            print(file)
            d = glob(os.path.join(file, "*.jpg"))
            data[feature].append(d)
    return data

def GetQuestionVectors(anchorImage_n):
    d1 = "RESULT/"
    return np.genfromtxt(d1 + "TrueValue/" + str(anchorImage_n) + ".txt")

def ProjectionOnQuestionVectors(data, questionVectors, data_truth, feature_number, n_bins):
    res = [[]for i in range(n_bins)]
    for b in range(n_bins):
        for d in data[feature_number][b]:
            res[b].append(np.dot(questionVectors[feature_number], data_truth[os.path.basename(d).split("_")[-1]]))
        res[b].sort()
        print(np.mean(res[b]))
        print(res[b])


image_number = 0
n_features_embedding = 1
n_features_truth = 10
n_bins = 2

questionVectors = GetQuestionVectors(image_number)
data_path_truth = "../RankSVM_DATA/Shoes/Supervised/result/"
data_truth = GetFeatures(data_path_truth, n_features_truth)

data_path_bin = "../RankSVM_DATA/Shoes/2/Histogramm/result"
data_bin = GetFeatures(data_path_bin, 10)

targert_image = random.choice(data_truth.keys())
print(targert_image)

for image_number in range(12):
    for feature_n in range(n_features_embedding):
        res = [[] for i in range(n_bins)]
        for b in range(n_bins):
            data_truth_bin_path = "RESULT/TrueValue/generated_" + str(image_number) + "_" + str(feature_n) + "_" + str(b) + "/"
            data_truth_bin = glob(os.path.join(data_truth_bin_path, "*.jpg"))
            data_index = 0
            for d in data_truth_bin:
                features = os.path.basename(d).split("_")[-n_features_truth-1:-1]
                features = np.array([float(f) for f in features])
                res[b].append(features)
                data_index += 1
        answers = []
        for d in range(data_index):
            answers.append([np.linalg.norm(data_truth[targert_image] - res[b][d]) for b in range(n_bins)])
        answers2 = np.array([float(a[0] > a[1] + 0.5) for a in answers])
        answers1 = np.array([float(a[0]+ 0.5 < a[1]) for a in answers])
        print(np.sum(answers1),np.sum(answers2))
        print(data_bin[targert_image])



