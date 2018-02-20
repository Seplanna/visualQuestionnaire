import numpy as np
from utils import GetData
import subprocess
import os
import argparse
import cv2

parser = argparse.ArgumentParser()

parser.add_argument('--check_embedding_for_heel_height', type = bool, default=False)
parser.add_argument('--check_embedding_for_the_length', type = bool, default=False)
parser.add_argument('--check_My_Embedding_For_the_Length', type = bool, default=False)
parser.add_argument('--split_images_by_category', type = bool, default=False)

parser.add_argument('--Brightness', type = bool, default=False)
parser.add_argument('--svm', default=False)



FLAGS, unparsed = parser.parse_known_args()

def ParceFile(data_file):
    line_n = 0
    dictionary = {'Category':{}, 'SubCategory':{}, 'HeelHeight':{}, 'Insole':{}, 'Closure':{},
                  'Gender': {}, 'Material' : {}, 'ToeStyle' : {}}
    res = {}
    n_features=0
    first_line = []
    with open(data_file) as f:
        for line in f:
            line = line.strip()
            if line_n == 0:
                n_features = len(line.split(',')) - 1
                first_line = line.split(",")[1:]
                print(line)

                #Heel Labels
                index = 0
                for l in first_line:
                    if l.startswith("HeelHeight"):
                        j = 5
                        while j > 0 and not l.startswith("HeelHeight." + str(j)):
                            j-=1
                        dictionary['HeelHeight'][l] = [index, j]
                    index += 1

                #Category
                index = 0
                j = 0
                for l in first_line:
                    if l.startswith("Category"):
                        dictionary['Category'][l] = [index, j]
                        j += 1
                    index += 1

                #SubCategory
                index = 0
                j = 0
                for l in first_line:
                    if l.startswith("SubCategory"):
                        dictionary['SubCategory'][l] = [index, j]
                        j += 1
                    index += 1

                #Gender
                dic = {'Gender.Men':0, 'Gender.Boys' : 0, 'Gender.Women' : 1, 'Gender.Girls' : 1, }
                index = 0
                for l in first_line:
                    if l.startswith("Gender"):
                        dictionary['Gender'][l] = [index, dic[l]]
                    index += 1

            if line_n > 0:
                line = line.split(",")
                res[line[0].replace("-", ".")] = np.array([float(l) for l in line[1:]])
            line_n+=1
    print (dictionary)
    return res, n_features, first_line, dictionary

def CheckProbabilityLabel(meta_data, cluster, n_features):
    res = np.zeros(n_features)

    data = GetData(cluster)
    for d in data:
        res += meta_data[os.path.basename(d).split("_")[-1][:-4]]

    return res

def GetLabelsToHeelLength(meta_data, data, dictionary, key):
    for k in dictionary.keys():
        dictionary[k][1] = 0
    dictionary[key][1] = 1


    #data format --- {name:score}
    res = []
    for d in data.keys():
        meta_features = meta_data[d]
        for f in dictionary.keys():
            if meta_features[dictionary[f][0]] > 0:
                res.append([d,data[d], dictionary[f][1]])
    return res

def QualityOfRanking(data, n_different_labels):
    # data format --- [[name, score,label]]

    res = 0.

    labels = np.zeros(n_different_labels)
    n_correct_pairs = 0.
    n_pairs = 0.
    data.sort(key=lambda x:-x[1])
    #print(data)
    for i in range(len(data)):
        #print(data[i], labels)
        label = int(data[i][2])
        for l in range(n_different_labels):
            if l < label:
                n_correct_pairs += labels[l]
            if not l==label:
                n_pairs += labels[l]
        labels[label] += 1.

    res = n_correct_pairs / n_pairs

    labels = np.zeros(n_different_labels)
    n_correct_pairs = 0.
    n_pairs = 0.
    data.sort(key=lambda x:x[1])
    #print(data)
    for i in range(len(data)):
        #print(data[i], labels)
        label = int(data[i][2])
        for l in range(n_different_labels):
            if l < label:
                n_correct_pairs += labels[l]
            if not l==label:
                n_pairs += labels[l]
        labels[label] += 1.

    #print(labels)
    if (n_correct_pairs / n_pairs > res):
        res = n_correct_pairs / n_pairs
    return res

def GetPicturesWithLabels(meta_data, data_path, dictionary, result_dir):
    print(dictionary)
    os.makedirs(result_dir)
    data = GetData(data_path)
    for d in data:
        basename = os.path.basename(d).split("_")
        meta_features = meta_data[basename[-1][:-4]]
        for f in dictionary.keys():
            if meta_features[dictionary[f][0]] > 0:
                subprocess.call(['cp', d, result_dir + basename[0] + "_" + str(dictionary[f][1]) + "_" + basename[-1]])






if FLAGS.check_embedding_for_heel_height:
    meta_data, n_features, first_line, dictionary = ParceFile("../CycleGAN_shoes/ut-zap50k-data/meta-data-bin.csv")
    print(dictionary["HeelHeight"])
    data_path = "../CycleGAN_shoes/DATA_ALL_LATENT1/"
    result_path = '../CycleGAN_shoes/heels_vector.txt'
    #GetPicturesWithLabels(meta_data, data_path + str(step) + "/", dictionary["HeelHeight"], "../CycleGAN_shoes/My/HeelHeight_" + str(step) + "/")



    vector = np.genfromtxt(result_path)
    vector = vector[-1] - vector[0]

    data_ = GetData(data_path)
    data = {}
    for d in data_:
        basename = os.path.basename(d).split("_")
        data[basename[-1][:-4]] = np.array([float(f) for f in basename[:-1]])

    d = dictionary["HeelHeight"]
    n_labels = 3
    res = []

    for d_ in data.keys():
        for k in d.keys():
            if meta_data[d_][d[k][0]] > 0:
                res.append([d, np.dot(vector, data[d_]),d[k][1] / 2])

    print(QualityOfRanking(res, 6))

if FLAGS.check_embedding_for_the_length:
    meta_data, n_features, first_line, dictionary = ParceFile("../CycleGAN_shoes/ut-zap50k-data/meta-data-bin.csv")
    dictionary = dictionary["SubCategory"]
    dictionary_ = {}
    dictionary_["SubCategory.Over.the.Knee"]=dictionary["SubCategory.Over.the.Knee"]
    dictionary_["SubCategory.Over.the.Knee"][1] = 3
    dictionary_["SubCategory.Knee.High"]=dictionary["SubCategory.Knee.High"]
    dictionary_["SubCategory.Knee.High"][1] = 2
    dictionary_["SubCategory.Mid.Calf"]=dictionary["SubCategory.Mid.Calf"]
    dictionary_["SubCategory.Mid.Calf"][1] = 1
    dictionary_["SubCategory.Ankle"]=dictionary["SubCategory.Ankle"]
    dictionary_["SubCategory.Ankle"][1] = 0
    data_path = "../CycleGAN_shoes/DATA_ALL_LATENT1/"
    result_path = '../CycleGAN_shoes/hight_vector.txt'
    
    n_labels=4
    
    data_ = GetData(data_path)
    data = {}
    for d in data_:
        basename = os.path.basename(d).split("_")
        data[basename[-1][:-4]] = np.array([float(f) for f in basename[:-1]])
    
    def Get_hight_vectors(dictionary, data, n_labels):
    
        n_features_embedding=10
        mean_vectors = np.zeros([n_labels, n_features_embedding])
        n_label=np.zeros(n_labels)
        for d_ in data.keys():
            for k in dictionary.keys():
                if meta_data[d_][dictionary[k][0]] > 0:
                    mean_vectors[dictionary[k][1]] += data[d_]
                    n_label[dictionary[k][1]] += 1.
        for i in range(n_labels):
            mean_vectors[i] /= n_label[i]
        return mean_vectors
    
    np.savetxt(result_path, Get_hight_vectors(dictionary_, data, n_labels))
    
    def CalculateRes(vector_path, data, dictionary, n_labels):
        vector = np.genfromtxt(vector_path)
        vector = vector[-1] - vector[0]
        res = []
    
        for d_ in data.keys():
            for k in dictionary.keys():
                if meta_data[d_][dictionary[k][0]] > 0:
                    res.append([d_, np.dot(vector, data[d_]),dictionary[k][1]])
    
        print(QualityOfRanking(res, n_labels))
    
    CalculateRes(result_path, data, dictionary_, n_labels)

if FLAGS.check_My_Embedding_For_the_Length:
    meta_data, n_features, first_line, dictionary = ParceFile("../CycleGAN_shoes/ut-zap50k-data/meta-data-bin.csv")
    dictionary = dictionary["SubCategory"]
    dictionary__ = {}
    dictionary__["SubCategory.Over.the.Knee"] = dictionary["SubCategory.Over.the.Knee"]
    dictionary__["SubCategory.Over.the.Knee"][1] = 3
    dictionary__["SubCategory.Knee.High"] = dictionary["SubCategory.Knee.High"]
    dictionary__["SubCategory.Knee.High"][1] = 2
    dictionary__["SubCategory.Mid.Calf"] = dictionary["SubCategory.Mid.Calf"]
    dictionary__["SubCategory.Mid.Calf"][1] = 1
    dictionary__["SubCategory.Ankle"] = dictionary["SubCategory.Ankle"]
    dictionary__["SubCategory.Ankle"][1] = 0

    vector = np.genfromtxt("../CycleGAN_shoes/Experiment/8_9.txt")

    dictionary__ = dictionary["HeelHeight"]

    n_labels = 3
    #data_path = "../CycleGAN_shoes/My/4/"
    data_path = "../CycleGAN_shoes/DATA_ALL_LATENT1/"

    res = []
    data = GetData(data_path)
    for d in data:
        basename = os.path.basename(d)
        d_ = basename.split("_")[-1][:-4]
        for k in dictionary__.keys():
            if meta_data[d_][0] > 0 and meta_data[d_][dictionary__[k][0]] > 0:
                features = np.array([float(f) for f in basename.split("_")[:-1]])
                res.append([d_, np.dot(features, vector), dictionary__[k][1] / 2])

    print(QualityOfRanking(res, n_labels))

if FLAGS.split_images_by_category:
    meta_data, n_features, first_line, dictionary = ParceFile("../CycleGAN_shoes/ut-zap50k-data/meta-data-bin.csv")

    n_categories = 4
    result_path = "../CycleGAN_shoes/DATA_ALL_LATENT1_categories/"
    data_path = "../CycleGAN_shoes/DATA_ALL_LATENT1/"

    os.makedirs(result_path)
    for i in range(n_categories):
        os.makedirs(result_path + str(i) + "/")

    data = GetData(data_path)
    for d in data:
        basename = os.path.basename(d)
        d_ = basename.split("_")[-1][:-4]
        for i in range(n_categories):
            if (meta_data[d_][i] > 0):
                subprocess.call(['cp', d, result_path + str(i) + "/" + basename])

if FLAGS.Brightness:
    data_path = "../CycleGAN_shoes/DATA_ALL_LATENT1/"
    svm_path=FLAGS.svm
    n_features=10
    svm = np.genfromtxt(svm_path)

    n_labels=5
    data = GetData(data_path)
    brightness = []
    for d in data:
        gray = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
        print(gray)
        basename = os.path.basename(d).split("_")[:n_features]
        brightness.append([os.path.basename(d), np.dot(svm, np.array([float(f) for f in basename])), np.mean(gray)])
    brightness.sort(key=lambda x:x[2])
    step = len(brightness) / n_labels
    for i in range(len(brightness)):
        brightness[i][2] = min(i / step, n_labels-1)

    brightness.sort(key=lambda x: x[1])
    print(QualityOfRanking(brightness, n_labels))











