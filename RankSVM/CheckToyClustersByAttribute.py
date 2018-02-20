from utils import GetData
import os
import numpy as np
from FeaturesFromCSV import QualityOfRanking
#--------------------------------------------------
def CheckClusterDistribution(data):
    n_features = 3
    res = np.zeros(n_features+1)
    for d in data:
        res += np.array([1] + [float(f) for f in os.path.basename(d).split("_")[:n_features]])
    return res

def CheckClusters(dataset, step):
    data1 = GetData(dataset + str(2*step) + "/")
    res_data1 = CheckClusterDistribution(data1)

    data2 = GetData(dataset + str(2*step + 1) + "/")
    res_data2 = CheckClusterDistribution(data2)

    print(res_data1, res_data2)
#--------------------------------------------------

def CheckRankingQuality(data_path, data_path_with_labels, feature_number):
    n_features = 3
    data_with_labels_ = GetData(data_path_with_labels)
    data_with_labels = {}
    for d in data_with_labels_:
        d = os.path.basename(d).split("_")
        data_with_labels[d[-1]] = [float(f) for f in d[:n_features]]

    data_ = GetData(data_path)
    data = []
    for d in  data_:
        basename = os.path.basename(d).split("_")
        data.append([basename[-1], float(basename[0]), data_with_labels[basename[-1]][feature_number]])
    print(QualityOfRanking(data, 2))





for i in range(3):
    print("step = ", i)
    CheckClusters("../CycleGAN_shoes/Toy/My/", i+1)
    for j in range(3):
        CheckRankingQuality("../CycleGAN_shoes/Toy/My_interpretability/" + str(i), "../CycleGAN_shoes/Toy/shoes_boots_heels_white_black/",j)