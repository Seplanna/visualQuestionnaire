import subprocess
import argparse
import os
import numpy as np
from glob import glob
from sklearn.cluster import KMeans
import random
from utils import Gist, GetData, get_image, imsave
from Evaluation import DevideFeatureIntoBins

parser = argparse.ArgumentParser()
parser.add_argument('--take_pictures_from_first_bin', type = bool, default=False)
parser.add_argument('--n_bins', type = int, default=-1)
parser.add_argument('--result_dir', default="")
parser.add_argument('--data_path', default="")

parser.add_argument('--check_ranking_for_the_clusters', type=bool, default=False)
parser.add_argument('--svm_model', default="")
parser.add_argument('--data_set', default="")
parser.add_argument('--step', default=-1,type=int)
parser.add_argument('--n_features_embedding', type = int, default=-1)

parser.add_argument('--create_real_pairs', type=bool, default=False)

parser.add_argument('--get_varietion', type=bool, default=False)
parser.add_argument('--create_clusters', type=bool, default=False)


FLAGS, unparsed = parser.parse_known_args()

# result_data_format: name of the file --- ****_f1_f2_f3_..._fn_file_name.jpg
def GetFeatures(svm_model, data_path, result_path, n_features):
    os.makedirs(result_path)
    svm = []
    for i in range(n_features):
        svm.append(np.genfromtxt(svm_model + str(i + 1) + ".txt"))
    data = glob(os.path.join(data_path, "*.jpg"))
    for d in data:
        g = Gist(d)
        res = [np.dot(svm[i], g) for i in range(n_features)]
        basename = os.path.basename(d)
        subprocess.call(["cp", d,
                         result_path + "/" + "_".join(
                             "{0:.2f}".format(r) for r in res[-1:]) + "_" +
                         basename])

# Get pairs from clusters
def ClusterData(data, embedding_size, n_clusters, result_dir, step):
    os.makedirs(result_dir)
    latent = np.zeros([len(data), embedding_size])
    for d in range(len(data)):
        basename = os.path.basename(data[d]).split("_")[-embedding_size-1:-1]
        latent[d] = np.array([float(f) for f in basename])
    clusters = KMeans(n_clusters=n_clusters, random_state=200).fit(latent)

    n_ex = 128
    images_indexes = []
    for c in range(n_clusters):
        cluster_images_index = [i for i in range(len(data)) if clusters.labels_[i] == c]
        images_indexes.append(cluster_images_index)
        print('cluster = ', c, len(cluster_images_index))
        distancse_to_cluster = clusters.transform(latent[cluster_images_index])
        distancse_to_cluster = distancse_to_cluster.T[c]
        arg_sort = np.argsort(distancse_to_cluster[:n_ex])
        images_index = np.array(cluster_images_index)
        im = images_index[arg_sort]
        batch_files = [data[i] for i in im]
        batch = [get_image(batch_file,
                               input_height=100,
                               input_width=150,
                               resize_height=100,
                               resize_width=150,
                               crop=False,
                               grayscale=False) for batch_file in batch_files]
        imsave(np.array(batch[:144]), [12, 12], result_dir + '/test_' + str(step) + '_' + str(c) + "_.png")
    return images_indexes, clusters

def GetTheClosestPoint(point, data, embedding_size):
    dist = 1e+10
    res=''
    for d in data:
        basename = os.path.basename(d).split("_")[-embedding_size-1:-1]
        local_dist = np.linalg.norm(point-np.array([float(f) for f in basename]))
        if local_dist < dist:
            dist=local_dist
            res=d
    return res

def BuildPairsFromRealImages(data_path1, data_path2, result, result_dir1, result_dir2):
    if not (os.path.exists(result)):
        os.makedirs(result)
    os.makedirs(result_dir1)
    os.makedirs(result_dir2)
    n_clusters = 10
    data1 = GetData(data_path1)
    random.shuffle(data1)
    data2 = GetData(data_path2)
    images_indexes1, clusters1 = ClusterData(data1, FLAGS.n_features_embedding, n_clusters, result + "0_test/", FLAGS.step)
    images_indexes2, clusters2 = ClusterData(data2, FLAGS.n_features_embedding, n_clusters, result + "1_test/", FLAGS.step)
    print("VARIETION")
    print(np.mean(clusters1.cluster_centers_, axis=0), np.mean(clusters2.cluster_centers_, axis=0))
    print(np.linalg.norm(clusters1.cluster_centers_ - np.mean(clusters1.cluster_centers_, axis=0)),
          np.linalg.norm(clusters2.cluster_centers_ - np.mean(clusters2.cluster_centers_, axis=0)))

    index = 0
    for cl in range(n_clusters):
        data1_ = [data1[i] for i in images_indexes1[cl]]
        im = GetTheClosestPoint(clusters1.cluster_centers_[cl], data1_, FLAGS.n_features_embedding)
        for cl1 in range(n_clusters):
            data2_ = [data2[i] for i in images_indexes2[cl1]]
            im1 = GetTheClosestPoint(clusters1.cluster_centers_[cl], data2_, FLAGS.n_features_embedding)
            subprocess.call(["cp", im, result_dir1 + str(index) + ".jpg"])
            subprocess.call(["cp", im1, result_dir2 + str(index) + ".jpg"])
            index += 1

    for cl in range(n_clusters):
        data2_ = [data2[i] for i in images_indexes2[cl]]
        im1 = GetTheClosestPoint(clusters2.cluster_centers_[cl], data2_, FLAGS.n_features_embedding)
        for cl1 in range(n_clusters):
            data1_ = [data1[i] for i in images_indexes1[cl1]]
            im = GetTheClosestPoint(clusters2.cluster_centers_[cl], data1_, FLAGS.n_features_embedding)
            subprocess.call(["cp", im, result_dir1 + str(index) + ".jpg"])
            subprocess.call(["cp", im1, result_dir2 + str(index) + ".jpg"])
            index += 1

if FLAGS.create_real_pairs:
    #n_features_embedding, step, result_dir
    step = FLAGS.step
    result_dir = FLAGS.result_dir
    data_path1 =  result_dir + str(2*step) + "/"
    data_path2 = result_dir + str(2*step + 1) + "/"

    result= result_dir + str(2*step) + "_" + str(2*step + 1) +"/"
    result_dir1 = result_dir + "test_AtoB_" +str(step) + "/"
    result_dir2 = result_dir + "test_BtoA_" +str(step) + "/"
    BuildPairsFromRealImages(data_path1, data_path2, result, result_dir1, result_dir2)



if FLAGS.check_ranking_for_the_clusters:
    def QualityBinaryClassifier(result):
        #result_format: [[score, truth]]
        result.sort(key=lambda x:x[0])
        result=np.array(result)
        n_ones_total = np.sum(result[:,1])
        print(result.shape, n_ones_total)
        binary_cls_result = 0.
        n_ones_current = 0.
        for i in range(result.shape[0]):
            if(result[i][1] > 0.5):
                n_ones_current += 1.
            current_quality = ((i+1) - n_ones_current) + (n_ones_total- n_ones_current)
            current_quality /= result.shape[0]
            if (current_quality > binary_cls_result):
                print(current_quality)
                binary_cls_result = current_quality
        return binary_cls_result


    #to run it should specify parameters: 1.svm_model
    #                                     2.result_dir
    #                                     3.data_set
    #                                     4.step
    #                                     5.n_features_embedding
    svm_model = FLAGS.svm_model#"../CycleGAN_shoes/My_"
    result_dir = FLAGS.result_dir
    data_path = FLAGS.data_set
    step = FLAGS.step
    data_path1 = data_path + str(2*(FLAGS.step-1)) + "/"
    data_path2 = data_path + str(2*(FLAGS.step-1) + 1) + "/"
    GetFeatures(svm_model, data_path1, result_dir + "0_", FLAGS.n_features_embedding)
    GetFeatures(svm_model, data_path2, result_dir + "1_", FLAGS.n_features_embedding)
    result = glob(os.path.join(result_dir+"0_/", "*.jpg"))
    data = []
    for d in result:
        basename = os.path.basename(d)
        data.append([-float(basename.split("_")[0]), 0.])

    result = glob(os.path.join(result_dir + "1_/", "*.jpg"))
    for d in result:
        basename = os.path.basename(d)
        data.append([-float(basename.split("_")[0]), 1.])

    print(QualityBinaryClassifier(data))





if FLAGS.take_pictures_from_first_bin:
    print("AAAAAAA")
    data_path=FLAGS.data_path
    result_dir=FLAGS.result_dir
    step = FLAGS.step

    n_bins=FLAGS.n_bins
    bins = DevideFeatureIntoBins(data_path, n_bins, 0)
    data = []
    for root, subdirs, files in os.walk(data_path):
        for f in files:
            if os.path.splitext(f)[1].lower() in ('.jpg', '.jpeg'):
                data += [os.path.join(root, f)]
    print(len(data))
    os.makedirs(result_dir + "_0/")
    for d in data:
        basename = os.path.basename(d)
        basename = basename.split("_")
        if (float(basename[0]) < bins[0]):
            subprocess.call(['cp', d, result_dir + "_0/" + "_".join(basename[1:])])

    os.makedirs(result_dir + "_1/")
    for d in data:
        basename = os.path.basename(d)
        basename = basename.split("_")
        if (float(basename[0]) > bins[1]):
            subprocess.call(['cp', d, result_dir + "_1/" + "_".join(basename[1:])])

    images_indexes1, clusters1 = ClusterData(GetData(result_dir + "_0/"), FLAGS.n_features_embedding, 20, result_dir + "_0_test/", FLAGS.step)
    images_indexes2, clusters2 = ClusterData(GetData(result_dir + "_1/"), FLAGS.n_features_embedding, 20, result_dir + "_1_test/", FLAGS.step)
    print("VARIETION")
    print(np.mean(clusters1.cluster_centers_, axis=0), np.mean(clusters2.cluster_centers_, axis=0))
    print(np.linalg.norm(clusters1.cluster_centers_ - np.mean(clusters1.cluster_centers_, axis=0)),
          np.linalg.norm(clusters2.cluster_centers_ - np.mean(clusters2.cluster_centers_, axis=0)))

if FLAGS.create_clusters:
    data_path=FLAGS.data_path
    result_dir=FLAGS.result_dir
    step = FLAGS.step
    data = GetData(data_path)
    images_indexes, clusters = ClusterData(data, FLAGS.n_features_embedding, 2, result_dir + str(2*step) + "_" + str(2*step+1), FLAGS.step)

    for d in range(len(images_indexes)):
        os.makedirs(result_dir + str(2*step+d))
        for d_ in images_indexes[d]:
            subprocess.call(["cp", data[d_], result_dir + str(2*step+d) + "/" + os.path.basename(data[d_])])

