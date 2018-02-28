"""
1.First Cluster all items into(10???) clusters and take the images near the centroids.
2.Devide each Feature into 5 bins
3. Ask user which of the images is more similar to the desiered one.
4. Take it and make one step for each direction, and take the clossest images (repeat steps 3 and 4)
"""

#Cluster ALL DATA
from glob import glob
import os
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn import svm, linear_model, cross_validation
from utils import imsave, get_image, Gist
import math
import sys

import subprocess
import argparse
from math import sqrt

"""parser = argparse.ArgumentParser()
parser.add_argument('--start', type = int)
parser.add_argument('--method')
FLAGS, unparsed = parser.parse_known_args()"""

def GetAnchorImagesFromAutoEncoderCluster(dirs, n_pictures_per_dir, data_path, result_dir):
    dir_n = 0
    for d in dirs:
        dir_n += 1
        data = glob(os.path.join(d, "*.jpg"))
        random.shuffle(data)
        for i in range(n_pictures_per_dir):
            name = os.path.basename(data[i]).split("_")[-1]
            subprocess.call(['cp', data_path[name], result_dir + str(dir_n) + "_" + name])


def Cluster(data, n_features, result_to_images_to_show,n_clusters, prefix):
    data = glob(os.path.join(data, "*.jpg"))
    random.shuffle(data)
    latent = np.zeros([len(data), n_features])
    for i in range(len(data)):
        try:
            latent[i] = np.array([float(i_) for i_ in os.path.basename(data[i]).split("_")[n_features:2*n_features]])
        except:
            print(os.path.basename(data[i]).split("_"))
            return
    clusters = KMeans(n_clusters=n_clusters, random_state=200).fit(latent)

    n_ex = 128
    images_indexes = []
    for c in range(n_clusters):
        images_index = [i for i in range(len(data)) if clusters.labels_[i] == c]

        print('cluster = ', c, len(images_index))
        distancse_to_cluster = clusters.transform(latent[images_index])
        distancse_to_cluster = distancse_to_cluster.T[c]
        arg_sort = np.argsort(distancse_to_cluster[:n_ex])
        images_index = np.array(images_index)
        images_indexes.append(images_index)
        im = images_index[arg_sort]
        result_images = [data[i] for i in im]
        batch = [get_image(batch_file,
                           input_height=64,
                           input_width=64,
                           resize_height=64,
                           resize_width=64,
                           crop=False,
                           grayscale=False) for batch_file in result_images]
        #imsave(np.array(batch), [12,12], prefix + str(c) + ".jpg")
        for ddd in range(3):
            subprocess.call(("cp " + data[im[ddd]] + " " + result_to_images_to_show + str(c) + "_" +
                         os.path.basename(data[im[ddd]]).split("_")[-1]).split())
        #os.makedirs(prefix + str(c))
        #for d in images_index:
        #    subprocess.call(["cp", data[d], prefix + str(c) + "/" + os.path.basename(data[d])])
    return images_indexes

def HierarhicalClustering(n_features, result_path,
                          result_to_images_to_show, data, images_indexes, prefix=""):
    #data = glob(os.path.join(data_path, "*.jpg"))
    if (len(prefix.split("_")) > 4 or images_indexes.shape[0] < 10):
        return
    n_clusters = 10
    data_n = [data[i] for i in images_indexes]
    print(len(data), prefix)
    images_indexes = Cluster(data_n, n_features, result_path, result_to_images_to_show, prefix, n_clusters)
    for i in range(n_clusters):
        new_prefix = str(i) + "_" + prefix
        HierarhicalClustering(n_features, result_path,
                          result_to_images_to_show, data, images_indexes[i], new_prefix)


def RunHierarhicalClustering(data_path, n_features,
                             result_path, result_to_images_to_show):
    data = glob(os.path.join(data_path, "*.jpg"))
    images_indexes = np.arange(len(data))
    HierarhicalClustering(n_features, result_path,
                          result_to_images_to_show, data, images_indexes, prefix="")


# Devide each Feature into 10 bins
def RecieveBinOfTheFeature(value, bins, n_bins):
    bin = 0
    while (bin < n_bins - 1 and value > (bins[bin])):
        bin += 1
    if (bin == n_bins):
        bin -= 1
    return bin

def MakeNormalFormat(data_path, result_file):
    data = glob(os.path.join(data_path, "*.jpg"))
    for d in data:
        name = os.path.basename(d).split("_")[5:]
        name = "_".join(name)
        subprocess.call(["cp", d, result_file + name])

def DevideFeatureIntoBinsFromData(values, n_bins):

    arg_sort = np.argsort(values)
    step = len(values) / n_bins
    values = np.array(values)
    bins = values[arg_sort[0::step]]
    bins = bins[1:]
    return bins

def DevideFeatureIntoBins(data_path, n_bins, feature_n):
    data = []
    for root, subdirs, files in os.walk(data_path):
        for f in files:
            #print(f)
            if os.path.splitext(f)[1].lower() in ('.jpg', '.jpeg'):
                data += [os.path.join(root, f)]
    values = [float(os.path.basename(i).split("_")[feature_n]) for i in data]
    return DevideFeatureIntoBinsFromData(values, n_bins)

def DevideFeaturesIntoBins(data_path, data_path_to_save, n_bins, n_features):
    if (not os.path.exists(data_path_to_save)):
        os.makedirs(data_path_to_save)
    #data_format: name of the file --- ****_f1_f2_f3_..._fn_file_name.jpg
    data = glob(os.path.join(data_path, "*.jpg"))
    data1 = {}
    for d in data:
        data1[os.path.basename(d).split("_")[-1]] = os.path.basename(d).split("_")[-n_features-1:-1]
    with open(data_path_to_save + "bins.txt", 'w') as bins:
        for i in range(n_features):
            b = DevideFeatureIntoBins(data_path, n_bins, -n_features - 1 + i)
            bins.write("\t".join(str(b_) for b_ in b))
            bins.write("\n")
            for i_ in range(len(data)):
                j = 0
                feature_value = float(os.path.basename(data[i_]).split("_")[-n_features - 1 + i])
                while j < n_bins-1 and b[j] < feature_value:
                    j+=1
                r = open(data_path_to_save + "result" + str(i) + "_" + str(j), 'a')
                data1[os.path.basename(data[i_]).split("_")[-1]].append(str(j))
                r.write(os.path.basename(data[i_]).split("_")[-1] + "\n")
    os.makedirs(data_path_to_save + "result/")
    for i_ in range(len(data)):
        basename = os.path.basename(data[i_]).split("_")[-1]
        d = [data1[basename][n_features-1-i] for i in range(n_features)]
        print(data1[basename])
        subprocess.call(["cp", data[i_], data_path_to_save + "result/" + "_".join(data1[basename][-1-i] for i in range(n_features)) + "_"
                 + "_".join(data1[basename][:n_features]) + "_" + basename])


#Choosing the question
def SatisfyConstraints(features, constraints):
    res = 0
    for i in constraints.keys():
        #print(i, features[i], constraints[i])
        if abs(features[i] -  constraints[i]) > 1e-10:
            return False
    return True


def FindTheClosestImage(data, data_bin, features, n_features, n_pictures=5, n_features_for_bin=0, constraints={}):
    res = []
    features = features[n_features:]
    for i in data:
        i = i.strip()
        if (SatisfyConstraints(data_bin[i][:n_features], constraints)):
            new_res = np.linalg.norm(features - data_bin[i][n_features_for_bin:])# - np.linalg.norm(features[feature_n] - data_bins[i][feature_n])
            res.append([i, new_res])
    res.sort(key=lambda x:x[1])
    return [i[0] for i in res[:n_pictures]]


def RecieveImageIndex(sorting, im_name):
    im_ind = 0
    while im_ind < len(sorting) and sorting[im_ind] != im_name:
        im_ind += 1
    return im_ind



def GetRanking(data_bins, features, im_name, n_features=5, n_picture=20, constraints={}):
    #SatisfyConstraints(data_bins[im_name][:n_features], constraints)
    res = [[im, np.linalg.norm(data_bins[im][n_features:] - features)] for im in data_bins.keys() if
           SatisfyConstraints(data_bins[im][:n_features], constraints)]
    res.sort(key=lambda x:x[1])
    del_ = [r[1] for r in res[:300]]
    #return del_
    #print(del_)
    im_ind = 0
    while im_ind < len(res) and res[im_ind][0] != im_name:
        im_ind += 1
    print("IM_IND, ", im_ind, len(res))
    return [r[0] for r in res[:n_picture]], im_ind

def GetQuestion(features, data_bins, n_bins, feature_n, path, noise, varietion, n_pictures_per_bin, n_features, constraints = {}, mode = 1):
    #print("n_bins =", n_bins)
    res_images = []
    i = feature_n
    if (noise):
        features = np.random.normal(features, varietion)
    for b in range(n_bins):
        data = open(path + "result" +str(i) + "_" + str(b)).read().splitlines()
        #data_bin_local = {}
        #for d in data:
        #    d = d.strip()
        #    if SatisfyConstraints(data_bins[d][:n_features], constraints):
        #        data_bin_local[d] = data_bins[d][n_features:]
        if (mode == 1):
            res = FindTheClosestImage(data, data_bins, features, n_features,
                                             n_pictures_per_bin, n_features, constraints)

        else:
            res = FindTheClosestImage(data, data_bins, features, n_features,
                                      n_pictures_per_bin, 5, constraints)

        res_images += res
    return res_images

def GetQuestionWithMaximumValue(features, data_bins, n_bins, feature_n, path, noise, varietion, n_pictures_per_bin, n_features, constraints = {}, mode = 1):
    #print("n_bins =", n_bins)
    res_images = []
    i = feature_n
    data = []
    print(data_bins[data_bins.keys()[0]])
    for d in data_bins.keys():
        if SatisfyConstraints(data_bins[d][:n_features], constraints):
            data.append([d, data_bins[d][-(i+1)]])
    data.sort(key=lambda x:x[1])

    step = len(data)/(n_bins-1)
    for b in range(n_bins-1):
        res_images += [d[0] for d in data[b*step:b*step + n_pictures_per_bin]]
        print("LEN RES IMAGES =", len(res_images))
    res_images += [d[0] for d in data[-n_pictures_per_bin:]]
    print("LEN RES IMAGES =", len(res_images))
    return res_images


def GetQuestionGenerated(features, data_bins, n_bins, feature_n, path, noise, varietion, n_pictures_per_bin, n_features, constraints = {}, mode = 1):
    data = glob(os.path.join("../CycleGAN-tensorflow/test__" + str(feature_n) + "_2", "*.jpg"))
    random.shuffle(data)
    data = data[:n_pictures_per_bin]
    data = [os.path.basename(d).split("_")[-1] for d in data]
    data_ = {}
    for d in range(len(data)):
        data_[data[d]] = d
    res_images = []

    res = ["" for i in range(n_pictures_per_bin)]
    data1 = glob(os.path.join("../CycleGAN-tensorflow/test_" + str(feature_n) + "_0", "*.jpg"))
    for d in data1:
        d_basename = os.path.basename(d).split("_")[-1]
        if d_basename in data:
            res[data_[d_basename]] = d
    res_images += res

    res = ["" for i in range(n_pictures_per_bin)]
    data1 = glob(os.path.join("../CycleGAN-tensorflow/test_" + str(feature_n) + "_1", "*.jpg"))
    for d in data1:
        d_basename = os.path.basename(d).split("_")[-1]
        if d_basename in data:
            res[data_[d_basename]] = d
    res_images += res

    res = ["" for i in range(n_pictures_per_bin)]
    data1 = glob(os.path.join("../CycleGAN-tensorflow/test__" + str(feature_n) + "_2", "*.jpg"))
    for d in data1:
        d_basename = os.path.basename(d).split("_")[-1]
        if d_basename in data:
            res[data_[d_basename]] = d
    res_images += res

    return res_images


def GetQuestionSave(features, data_path, data_bin, n_bins, feature_n, path, noise, varietion, n_features):

    res = []
    i = feature_n
    if (noise):
        features = np.random.normal(features, varietion)
    for b in range(n_bins):
        data = open(path + "/result" + str(i) + "_" + str(b)).readlines()
        r = GetExampleInBin(data, data_bin, data_path, features, n_features, str(b) + ".jpg")
        #os.makedirs(str(i) + "_" + str(b))
        #for r_ in r:
        #    subprocess.call(["cp", data_path[r_], str(i) + "_" + str(b) + "/" + os.path.basename(r_).split("_")[-1]])
        res.append(r)
    return res

def GetExampleInBin(data, data_bin, data_path, features, n_features,path):
    res = FindTheClosestImage(data, data_bin, features, n_features, 128, n_features)
    images = [get_image(data_path[im], 64, 64,
              resize_height=64, resize_width=64) for im in res]
    imsave(np.array(images), [12,12], path)
    return res

#Try to use CycleGan
def GetQuestionMyOneDirection(anchor_ims, checkpoint_dir, data_bin, data_path, n_features,
                              direction):
    print(anchor_ims)
    dir_ = "/Users/admin/Desktop/UVA/third/CycleGAN-tensorflow/"
    for i in glob(os.path.join("result/", "*.jpg")):
        subprocess.call(["rm", i])
    for anchor_im in anchor_ims:
        subprocess.call(["cp", data_path[anchor_im], "result/"])
    for i in glob(os.path.join(dir_ + "test1/", "*.jpg")):
        subprocess.call(["rm", i])
    #for i in glob(os.path.join(dir_ + "test/", "*.jpg")):
    #    subprocess.call(["rm", i])

    subprocess.call(["python", dir_ + "main.py",
                     "--dataset_a", "result/",
                    "--dataset_b", "result/",
                     "--which_direction", direction,
                     "--phase",  "test",
                     "--checkpoint_dir", checkpoint_dir,
                     "--test_dir", dir_ + "test/",
                     "--test_dir_a", dir_ + "test1/",
                     "--test_dir_b", dir_ + "test/"])
    images = []
    file = glob(os.path.join(dir_ + "test1/", "*.jpg"))
    for f in file:
        gist = Gist(f)
        svm = []
        for i in range(n_features):
            svm.append(np.genfromtxt("../CycleGAN_shoes/Black/" + str(i + 1) + ".txt"))
        res = np.array([np.dot(svm[i], gist) for i in range(n_features)])
        print(res)
        images+= FindTheClosestImage(data_bin.keys(), data_bin, res, 0, 1, n_features)
    return images

def GetQuestionMy(anchor_im, checkpoint_dir, data_bin, data_path, n_pictures_per_bin, n_features):
    anchor_ims_features = np.random.normal(data_bin[anchor_im][n_features:], 0.5,
                                           [n_pictures_per_bin, n_features])
    anchor_ims = []
    for i in range(n_pictures_per_bin):
        anchor_ims += FindTheClosestImage(data_bin.keys(), data_bin, anchor_ims_features[i], 0, 1, n_features)
    res = GetQuestionMyOneDirection(anchor_ims, checkpoint_dir, data_bin, data_path, n_pictures_per_bin, n_features,
                              "AtoB")
    res += anchor_ims
    res += GetQuestionMyOneDirection(anchor_ims, checkpoint_dir, data_bin, data_path, n_pictures_per_bin, n_features,
                              "BtoA")
    print(res)
    return res
    

def GetQuestionWithHierarhicalClustering(chosen_picture, data_cluster, n_steps, n_bins):
    for i in data_cluster[n_steps].keys():
        if (data_cluster[n_steps][i] == chosen_picture):
            chosen_picture1 = i

    data = data_cluster[n_steps+1]
    new_question = chosen_picture1
    res_images = []
    for i in range(n_bins):
        try:
            res_images.append(data[new_question + "_" + str(i)])
        except:
            continue
    return res_images

#Recieve the answer
def Answer(targert, shown, data_truth):
    res = ''
    res_dist = 1e+10
    ind_res = 0
    ind = 0
    res_all =[]
    for im in shown:
        res_local = np.linalg.norm(data_truth[targert] - data_truth[im])
        res_all.append(res_local)
        if (res_local < res_dist):
            res_dist = res_local
            res = im
            ind_res = ind
        ind += 1
    return res, res_dist

#OverallSystem
def ChooseTwoQuestionsToAnswer(data_bin, data_truth, data_path_histogramm,
                 chosen_picture, n_bins, anchor_features, n_features):
    n_pictures_per_bin = 5
    res = []

    for f in range(n_features):
        shown = GetQuestion(anchor_features, data_bin, n_bins, f, data_path_histogramm, False, 0.,
                    n_pictures_per_bin,
                    0, {}, 2)
        dis = []
        for b in range(n_bins):
            d = 0
            for p in range(n_pictures_per_bin):
                d += np.linalg.norm(data_truth[shown[n_pictures_per_bin*b + p]] - data_truth[chosen_picture])
            dis.append(d)
        dis = np.array(dis)
        ar_s = np.argsort(dis)
        res.append([dis[ar_s[1]] - dis[ar_s[0]], f, ar_s[0], dis[ar_s[1]], dis[ar_s[0]]])

    print([res[n_features-1-i][2] for i in range(n_features)], data_bin[chosen_picture][:n_features])
    #print("Prediction All, Real = ", res, data_bin[chosen_picture])
    #res.sort(key=lambda x:-x[0])
    print("Prediction, Real = ", res[0][2], data_bin[chosen_picture][n_features - 1 - res[0][1]])
    r = [int(data_bin[chosen_picture][n_features - 1 - res[i][1]] == res[i][2]) for i in range(n_features)]
    #r[0] = r[0]*r[1]
    return np.array(r)

def AnswerOnOneQuestion(bin_truth, truth, data_path_bin, data_path_truth, n_bins, n_features_truth):
    data_My = glob(os.path.join("resultMy_", "*.jpg"))
    res = [[float(os.path.basename(d).split("_")[0]), os.path.basename(d).split("_")[1]] for d in data_My]
    res.sort(key=lambda x:-x[0])
    n_p  = 30
    names = [res[i][1] for i in range(n_p)]

    n_pictures_per_bin = 128
    res = []
    data_bin = glob(os.path.join(data_path_bin, "*.jpg"))
    print(data_path_bin, len(data_bin))
    data_truth_ = glob(os.path.join(data_path_truth, "*.jpg"))
    data_truth = {}
    for d in data_truth_:
        name = os.path.basename(d).split("_")
        if name[-1] not in names:
            continue
        if(name[-1]) not in data_truth:
            data_truth[name[-1]] = [[] for r in range(n_bins)]
        data_truth[name[-1]][int(name[-2])-1] = np.array([float(f) for f in name[:n_features_truth]])

    dis = [0. for i in range(n_bins)]
    dis__ = []
    for d in data_truth.keys():
        dis_ = []
        for b in range(n_bins):
            dis_.append([np.linalg.norm(data_truth[d][b] - truth), b])
        dis_.sort(key=lambda x:x[0])
        dis__.append([dis_[1][0] - dis_[0][0], dis_[0][1]])
    dis__.sort(key=lambda x:-x[0])
    #print(dis__)
    for i in range(n_p):
        dis[dis__[i][1]] += 1
    print(int(np.argmax(dis) == bin_truth[4]))
    return int(np.argmax(dis) == bin_truth[4])


def OneIteration(data_, data_bin, data_truth, data_path_histogramm,
                 chosen_picture, n_bins, anchor_features, n_features, n_trials, varietion_coef=3., noise=False, Method=False):

    im_inds = []
    anchor_im = FindTheClosestImage(data_bin.keys(), data_bin, anchor_features, 0, 2, n_features)[0]
    del_, im_ind = GetRanking(data_truth, data_truth[chosen_picture],anchor_im, 0, len(data_truth.keys()))
    #print ("IM IND FIRST =", im_ind)
    im_inds.append(im_ind)

    anchor_features = data_bin[anchor_im].copy()
    res_p = np.linalg.norm(data_truth[chosen_picture] - data_truth[anchor_im] + 1e-10)
    n_pictures_per_bin = 3
    n_shows = 0
    for n_iterations in range(n_trials):
        features = range(n_features)
        random.shuffle(features)
        for i in features:
            if(n_shows >= n_trials * n_features):
                break
            n_shows += 1
            noise = noise
            res_p = np.linalg.norm(data_truth[chosen_picture] - data_truth[anchor_im])
            varietion = (res_p + 1e-10) / varietion_coef
            if (Method == 'Random'):
                shown = GetRandomPicture(data_, n_bins * n_pictures_per_bin)
                #shown = GetRandomPicture(data_, n_bins*n_pictures_per_bin)
                #shown = GetRandomPictureAnchor(data_bin, data_bin[anchor_im], n_bins * n_pictures_per_bin,
                #                               np.linalg.norm(data_truth[chosen_picture] - data_truth[anchor_im] + 1e-10))
            #elif (Method=='My'):
            #    shown = GetQuestionMy(anchor_im, "../CycleGAN_shoes/Black/checkpoint" + str(i+1) + "/"
            #                  , data_bin, data_path, n_pictures_per_bin, n_features)
            else:
                shown = GetQuestion(anchor_features, data_bin, n_bins, i, data_path_histogramm, noise, varietion,
                    n_pictures_per_bin,
                        n_features)


            res, res_dist = Answer(chosen_picture, shown, data_truth)
            if (res_p > res_dist):
                anchor_features += (data_bin[res] - anchor_features)
                #anchor_im = FindTheClosestImage(data_bin.keys(), data_bin, anchor_features, 0, 2)
                #anchor_im = anchor_im[0]
                anchor_im = res
                im_ind = RecieveImageIndex(del_, anchor_im)
                #print ("IMG_IND= ", im_ind)
                #del_, im_ind = GetRanking(data_truth, data_truth[chosen_picture], anchor_im, 0)
            #else:
            #    anchor_features += (data_bin[res] - anchor_features)*(1./math.exp(-res_p + res_dist))*res_p / 10



            im_inds.append(im_ind)
            #if (res_p > res_dist):
            #    anchor_im = res
            #    res_p = res_dist
            """else:
                shown = GetRandomPictureAnchor(data_bin, data_bin[anchor_im], n_bins * n_pictures_per_bin,
                                               np.linalg.norm(data_truth[chosen_picture] - data_truth[anchor_im] + 1e-10))
                res, res_dist = Answer(chosen_picture, shown, data_truth)
                if (res_p > res_dist):
                    anchor_im = res
                    res_p = res_dist
                n_shows += 1"""




        #print("RESULT = ", res_p, "N_FEATURES = ", n_features, n_shows)
        result = anchor_features[n_features:]
    res, im_ind = GetRanking(data_truth, data_truth[chosen_picture], anchor_im,0, 20)
    res1, im_ind1 = GetRanking(data_bin, data_bin[chosen_picture], anchor_im, 0, 20)
    res = res[:20]
    #print([np.linalg.norm(data_truth[r] - data_truth[chosen_picture]) for r in res])
    anchor_im = res[0]
    return res, im_inds,im_ind1

def GetRandomPicture(data_, n_pictures):
    chosen_pictures = []
    for i in range(n_pictures):
        chosen_picture1 = random.choice(data_)
        chosen_picture1 = os.path.basename(chosen_picture1).split("_")[-1]
        chosen_pictures.append(chosen_picture1)
    return chosen_pictures

def GetAnchorPictures(data_path):
    data = glob(os.path.join(data_path, "*.jpg"))
    chosen_pictures = []
    for i in data:
        chosen_picture1 = i.split("_")[-1]
        chosen_pictures.append(chosen_picture1)
    return chosen_pictures

def GetFirstIterationWithQuestionnaire(path_questionnare, data_truth, chosen_picture, n_features_truth):
    questions = glob(os.path.join(path_questionnare, "*.jpg"))
    n_questions = 4
    feature_truth = data_truth[chosen_picture]
    result = []
    for i in range(n_questions):
        image1 = [im for im in questions if (im.find(str(i) + "_truth.jpg") >= 0)][0]
        image2 = [im for im in questions if (im.find(str(i) + "_fake.jpg")>=0)][0]
        dis_im1 = np.linalg.norm(np.array([float(f) for f in os.path.basename(image1).split("_")[:n_features_truth]])
                                 -feature_truth)
        dis_im2 = np.linalg.norm(np.array([float(f) for f in os.path.basename(image2).split("_")[:n_features_truth]])
                                 -feature_truth)
        #print(dis_im1, dis_im2)
        #print(image1, image2)
        if (dis_im1 < dis_im2):
            result.append(float(os.path.basename(image1).split("_")[n_features_truth+i]))
        else:
            result.append(float(os.path.basename(image2).split("_")[n_features_truth + i]))
    result.append(float(os.path.basename(random.choice(questions)).split("_")[n_features_truth + n_questions]))
    return np.array(result)


def CreatAllData(data_path_bin, data_path_truth, n_features):
    n_features_truth = 10
    data_ = glob(os.path.join(data_path_bin, "*.jpg"))
    data_bin = {}
    data_path = {}
    for i in data_:
        basename = os.path.basename(i)
        data_bin[basename.split("_")[-1]] = np.array([float(f) for f  in basename.split("_")[:2*n_features]])
        data_path[basename.split("_")[-1]] = i


    data1_ = glob(os.path.join(data_path_truth, "*.jpg"))
    data_truth = {}
    for i in data1_:
        basename = os.path.basename(i)

        data_truth[basename.split("_")[-1]] = np.array([float(f) for f  in basename.split("_")[:n_features_truth]])
    return data_bin, data_path, data_truth, data_


def RunEvaluation(data_, data_bin, data_path, data_truth, data_path_questions, data_path_histogramm,
                  data_test, data_path_anchor_images,
                  n_features, n_bins, trial=0, n_trials=2,  varietion_coef = 3., Method='My'):



    #random.shuffle(data_test)
    chosen_picture = data_test[trial]
    chosen_picture = os.path.basename(chosen_picture).split("_")[-1]
    #subprocess.call(["cp", data_path[chosen_picture], "result1/0" + "_" + chosen_picture])


    im_inds = [0]
    im_ind1 = 0
    res,res1 =0,0
    for i in range(1):
        chosen_pictures = GetAnchorPictures(data_path_anchor_images)

        anchor_im, ind_res = Answer(chosen_picture, chosen_pictures, data_truth)
        anchor_features = data_bin[anchor_im].copy()[n_features:]

        chosen_pictures,im_inds,im_ind1 = OneIteration(data_, data_bin, data_truth, data_path_histogramm,
                 chosen_picture, n_bins, anchor_features, n_features, n_trials, varietion_coef, True, Method)

        res = ChooseTwoQuestionsToAnswer(data_bin, data_truth, data_path_histogramm,
                                   chosen_picture, n_bins, data_bin[anchor_im].copy()[n_features:], n_features)

        #res1 = AnswerOnOneQuestion(data_bin[chosen_picture], data_truth[chosen_picture], "resultMy/", "result_Supervised/", n_bins, 10)
        #print("RES ", res)


        #print("RESULT = ", np.linalg.norm(data_truth[chosen_pictures[0]] - data_truth[chosen_picture]))

    ind = 0
    real_dist = []
    for r in chosen_pictures:
        real_dist.append(np.linalg.norm(data_truth[r] - data_truth[chosen_picture]))
        ind += 1
    #print(data_bin[chosen_picture][:5])
    return real_dist[0], im_inds, im_ind1, res


def RunEvaluationWithAnchorImages(data_path_bin, data_path_truth, data_path_questions, n_bins):
    import shutil
    #shutil.rmtree('result/')
    shutil.rmtree('result1/')
    shutil.rmtree('result2/')
    #os.makedirs('result')
    os.makedirs('result1')
    os.makedirs('result2')
    n_features = 5
    n_features_truth = 10
    data_ = glob(os.path.join(data_path_bin, "*.jpg"))
    data_bin = {}
    data_path = {}
    for i in data_:
        basename = os.path.basename(i)
        data_bin[basename.split("_")[-1]] = np.array([float(f) for f  in basename.split("_")[:n_features]])
        data_path[basename.split("_")[-1]] = i


    data1_ = glob(os.path.join(data_path_truth, "*.jpg"))
    data_truth = {}
    for i in data1_:
        basename = os.path.basename(i)
        data_truth[basename.split("_")[-1]] = np.array([float(f) for f  in basename.split("_")[:n_features_truth]])

    chosen_picture = random.choice(data_)
    chosen_picture = os.path.basename(chosen_picture).split("_")[-1]
    subprocess.call(["cp", data_path[chosen_picture], "result1/0" + "_" + chosen_picture])

    result = []
    for i in range(n_bins):
        res = 1e+10
        res_f = 0
        questions_ = glob(os.path.join(data_path_questions + str(i+1) + "/", "*.jpg"))
        for q in questions_:
            bin_features = [float(f) for f in os.path.basename(q).split("_")[:n_features]]
            truth_features = [float(f) for f in os.path.basename(q).split("_")[n_features:n_features+n_features_truth]]
            res_local = np.linalg.norm(np.array(bin_features) - data_bin[chosen_picture])
            print(res_local, bin_features[i])
            if (res_local < res):
                res = res_local
                res_f = bin_features[i]
        result.append(res_f)

    print(data_bin[chosen_picture], result)
    res = GetRanking(data_bin, result, chosen_picture)
    ind = 0
    real_dist = []
    subprocess.call(["cp", data_path[chosen_picture], "result2/0" + "_" + chosen_picture])
    for r in res:
        subprocess.call(["cp", data_path[r], "result2/" + str(ind + 1) + "_" + r])
        real_dist.append(np.linalg.norm(data_truth[r] - data_truth[chosen_picture]))
        ind += 1
    print(real_dist)












#Last step
def GetQuestionForQuestionnaireOneBin(files):
    data = set()
    all_data = []
    for f in range(len(files)):
        d = set(os.path.basename(i).split("_")[-1] for i in glob(os.path.join(files[f], "*.jpg")))
        print("DDDDD =", d, files[f])
        all_data_file = {}
        for f1 in glob(os.path.join(files[f], "*.jpg")):
            all_data_file[os.path.basename(f1).split("_")[-1]] = f1
        all_data.append(all_data_file)
        if (f == 0):
            data = d
        else:
            data = data.intersection(d)




    res = 0
    res_im = ""
    to_copy = []
    for d in data:
        res_local = 0
        for f in range(len(files)):
            res_local += float(os.path.basename(all_data[f][d]).split("_")[0])
        if (res < res_local):
            res = res_local
            res_im = d
    if res == 0:
        return [res, res_im, to_copy]
    for f in range(len(files)):
        to_copy.append(all_data[f][res_im])
    print(res)
    return [res, res_im, to_copy]


def GetRandom(data_path, n_ex, test_path):
    data = glob(os.path.join(data_path, "*.jpg"))
    random.shuffle(data)
    for i in range(n_ex):
        subprocess.call(["cp", data[i], test_path + os.path.basename(data[i])])

def GetRandomPictureAnchor(data_bins, features, n_ex, varietion):
    res = []

    for i in range(n_ex):
        features = np.random.normal(features, varietion)
        res.append(FindTheClosestImage(data_bins.keys(), data_bins, features, 0, 1)[0])
    return res


def CalculateProbabilityOfFeatures(path, n_features, n_bins):
    data = glob(os.path.join(path, "*.jpg"))
    res = np.zeros([n_features, n_bins])
    for i in data:
        i = os.path.basename(i).split("_")[:n_features]
        for j in range(n_features):
            res[j][float(i[j])] += 1
    return res / len(data)




def SaveImagesNearAnchor(anchor_ims_path, data_path_bin, n_bins, feature_n, path, result_dir, varietion, n_features):
    data_ = glob(os.path.join(data_path_bin, "*.jpg"))
    data_bin = {}
    data_path = {}
    for i in data_:
        basename = os.path.basename(i)
        data_bin[basename.split("_")[-1]] = np.array([float(f) for f  in basename.split("_")[:2*n_features]])
        data_path[basename.split("_")[-1]] = i

    anchor_ims = glob(os.path.join(anchor_ims_path, "*.jpg"))
    for i in range(len(anchor_ims)):
        features = data_bin[os.path.basename(anchor_ims[i]).split("_")[-1]]
        res = GetQuestionSave(features, data_path, data_bin, n_bins, feature_n, path, False, varietion, n_features)
        for r in range(len(res)):
            os.makedirs(result_dir + str(i) + "_" + str(feature_n) +"_"+str(r))
            for j in res[r]:
                subprocess.call(["cp", data_path[j], result_dir + str(i) + "_"+ str(feature_n) +"_" +str(r) + "/" + os.path.basename(j)])

def Translate(data_path, feature, n_bins):
    for bin in range(n_bins):
        data = glob(os.path.join(data_path + str(feature) + "_" + str(bin) + "/", "*.jpg"))
        with open("result" + str(feature) + "_" + str(bin), "w") as res:
            for f in data:
                res.write(os.path.basename(f).split("_")[-1] + "\n")

def TranslateAll(data_path, n_bins, n_features):
    for feature in range(n_features):
        Translate(data_path, feature, n_bins)

def TranslateFromResult(path, n_features):
    data = glob(os.path.join(path, "*.jpg"))
    for d in data:
        basename = os.path.basename(d)
        basename = basename.split("_")
        for feature in range(n_features):
            with open("result" + str(feature) + "_" + basename[n_features - 1 - feature], "a") as r:
                r.write(basename[-1] + "\n")


#Check Supervised Embedding
#TranslateFromResult("../RankSVM_DATA/Shoes/Autoencoder/result9/", 10)
#TranslateAll("../RankSVM_DATA/Shoes/Autoencoder/result", 3, 10)
#Cluster("../RankSVM_DATA/Shoes/Histogramm/result4/",5, "Anchor_im/", 15, "Cluster/")

class Parameters():
    def __init__(self, method):
        self.data_path_truth = "../RankSVM_DATA/Shoes/Supervised/result/"

        if method == 'AutoEncoder':
            self.data_path_bin = "../RankSVM_DATA/Shoes/Autoencoder/result9/"
            self.n_features = 10
            self.data_path_histogramm = "../RankSVM_DATA/Shoes/Autoencoder/"

        if method == "My":
            self.data_path_bin = "../RankSVM_DATA/Shoes/Histogramm/result4/"
            self.n_features = 5
            self.data_path_histogramm = "../RankSVM_DATA/Shoes/Histogramm/"

        if method == "Random":
            self.data_path_bin = "../RankSVM_DATA/Shoes/Histogramm/result4/"
            self.n_features = 5
            self.data_path_histogramm = "../RankSVM_DATA/Shoes/Histogramm/"


if __name__ == "__main__":
    #GetRandom("../RankSVM_DATA/Shoes/Histogramm/result4/", 1000, "Test/")

    varietion_coef_beg = 1.5
    result_min = 1.
    best_var_coef = varietion_coef_beg
    n_examples = 1000
    n_questions = 10
    res_ = np.zeros(5)
    for v in range(1):
        res = 0
        map = [0 for i in range(n_questions)]
        map1 = 0.
        varietion_coef = varietion_coef_beg + 0.5*v

        params = Parameters(FLAGS.method)
        data_bin, data_path, data_truth, data_ = CreatAllData(params.data_path_bin,
                                                              params.data_path_truth,
                                                              params.n_features)

        chosen_pictures = GetAnchorPictures("Anchor_im1/")
        anchor_im = chosen_pictures[4]



        data_test = (glob(os.path.join("Test/", "*.jpg")))
        random.shuffle(data_test)

        res1 = np.zeros(300)
        for i_ in range(n_examples):
            i = i_ + FLAGS.start
            print("STEP = ", i)
            method = FLAGS.method
            n_trial = n_questions / params.n_features
            chosen_picture = os.path.basename(data_test[i]).split("_")[-1]
            #subprocess.call(["cp", data_test[i], "DEL/" + str(i) + ".jpg"])
            #res1 += np.array(GetRanking(data_truth, data_truth[chosen_picture], anchor_im, 0, len(data_truth.keys())))
        #print(res1/n_examples)
            res1, im_inds, im_ind1, res1_ = \
                    RunEvaluation(data_, data_bin, data_path, data_truth,
                              "result/", params.data_path_histogramm,
                              data_test, "Anchor_im1/",
                              params.n_features, 3, i, n_trial, varietion_coef, method)


            res+=res1
            res_ += res1_
            for q in range(n_questions):
                map[q] += 1./(im_inds[q]*0.05+1)
            map1 += 1./(im_ind1*0.05 + 1)
            if (i%100 == 0):
                print(i)
                map_ = np.array(map)
                np.savetxt(method + "_" + str(FLAGS.start) + "_" + str(i+1) + ".txt", map_ / (i_+1))
                print("RESULT = ,", np.array(map1) / n_examples, np.array(res) / (i_+1), np.array(map) / (i_+1), res_)

    #print("RESULT = ," ,np.array(map1)/n_examples, np.array(res) / n_examples, np.array(map) / n_examples, res_)










"""dir = "../CycleGAN_shoes/"
params = Parameters(FLAGS.method)
data_bin, data_path, data_truth, data_ = CreatAllData(params.data_path_bin,
                                                      params.data_path_truth,
                                                      params.n_features)
GetAnchorImagesFromAutoEncoderCluster([dir + "1_/", dir + "3_/", dir + "4_/", dir + "6_/", dir + "8_/"], 3, data_path, "Anchor_im1/")"""

###Plan
#1.Save parameters of SVM
#2.Make automative these CycleGAN (include confedence in test files)
#3.Predict theier value and make evaluation
