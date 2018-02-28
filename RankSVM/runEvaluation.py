import subprocess
import argparse
import os
from glob import glob
import shutil
import numpy as np
import random
from PIL import Image

from utils import Gist, GetHistogramm, chi2_distance, ResizeImage

parser = argparse.ArgumentParser()
parser.add_argument('--start', type = int, default=-1)
parser.add_argument('--n_features_truth', type = int, default=-1)
parser.add_argument('--n_features_embedding', type = int, default=-1)
parser.add_argument('--n_bins', type = int, default=-1)
parser.add_argument('--n_anchor_images', type=int, default=-1)
parser.add_argument('--method', default="My")
parser.add_argument('--run_bin', type=bool, default=False)
parser.add_argument('--run_generate_CycleGANData', type=bool, default=False)
parser.add_argument('--run_generate_images', type=bool, default=False)
parser.add_argument('--get_features_for_generated_images', type=bool, default=False)
parser.add_argument('--get_Myfeatures_for_real_images', type=bool, default=False)
parser.add_argument('--calculate_colour_histogramm_similarity', type=bool, default=False)
parser.add_argument('--check_the_mean_distance_generated_images', type=bool, default=False)
parser.add_argument('--save_random_examples', type=bool, default=False)
parser.add_argument('--calculate_colour_histogramm_similarity_for_generated_images', type=bool, default=False)
parser.add_argument('--calculate_colour_histogramm_similarity_for_real_images', type=bool, default=False)
parser.add_argument('--delete_white_border', type=bool, default=False)
parser.add_argument('--histogramm_classifier', type=bool, default=False)
parser.add_argument('--take_pictures_from_first_bin', type=bool, default=False)

parser.add_argument('--data_set', default="")
parser.add_argument('--result_dir', default="")
parser.add_argument('--n_examples', type = int, default=-1)




FLAGS, unparsed = parser.parse_known_args()



from Evaluation import DevideFeaturesIntoBins, SaveImagesNearAnchor, DevideFeatureIntoBins

def SaveImagesWithoutWhiteBorder(data_path, result_dir):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    data = glob(os.path.join(data_path, "*.jpg"))
    for d in data:
        try:
            im = ResizeImage(d)
            basename = os.path.basename(d)
            im = Image.fromarray(im)
            im.save(result_dir+basename)
        except:
            continue


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

#Check the system
def GetRandomExamples(data_set_path, n_examples):
    data = glob(os.path.join(data_set_path, "*.jpg"))
    print(data_set_path, len(data))
    examples = random.sample(data, n_examples)
    return examples

def SaveExamples(examples, result_dir, feature_n):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    for ex in examples:
        basename = os.path.basename(ex).split("_")
        if feature_n > 0:
            to_write = basename[feature_n] + "_"
        else:
            to_write = ''
        subprocess.call(["cp", ex, result_dir + to_write + os.path.basename(ex)])

def SaveRandomExamples(dataset_path, n_examples, result_dir, feature_number):
    SaveExamples(GetRandomExamples(dataset_path, n_examples), result_dir, feature_number)

if FLAGS.save_random_examples:
    SaveRandomExamples(FLAGS.data_set, FLAGS.n_examples, FLAGS.result_dir, FLAGS.n_features_embedding)
#Generate Bin
if FLAGS.run_bin:
    DevideFeaturesIntoBins("../RankSVM_DATA/Shoes/Histogramm/result4/", "../RankSVM_DATA/Shoes/2/Histogramm/",
                           FLAGS.n_bins, FLAGS.n_features_embedding)

#GenerateData For CycleGAN
if FLAGS.run_generate_CycleGANData:
    #dataFormat dataPath_bin
    result_dir = "../RankSVM_DATA/Shoes/2/Histogramm/CycleGan/DATA/"
    anchor_images = "Anchor_im1/"
    data = "../RankSVM_DATA/Shoes/2/Histogramm/result/"
    data_bin = "../RankSVM_DATA/Shoes/2/Histogramm/"
    n_bins = FLAGS.n_bins
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    for i in range(FLAGS.n_features_embedding):
        SaveImagesNearAnchor(anchor_images, data, n_bins, i,
                             data_bin, result_dir, 0, FLAGS.n_features_embedding)

#Generate Generated Images
if FLAGS.run_generate_images:
    #dataFormat dataPath_bin
    def GenerateImages2bins(data_paths, checkpoint_dir, result_path, test_dir, direction):
        dir_ = "/Users/admin/Desktop/UVA/third/CycleGAN-tensorflow/"
        opposite_direction = "BtoA" if direction=="AtoB" else "AtoB"
        del_dirs = ["del/", "del1/"]
        os.makedirs(del_dirs[0])
        os.makedirs(del_dirs[1])
        del_dir,del_dir1 = (del_dirs[0], del_dirs[1]) if direction=="AtoB" else (del_dirs[1], del_dirs[0])
        res0,res1 = (result_path[0], result_path[1]) if direction=="AtoB" else (result_path[1], result_path[0])
        if (direction == "AtoB"):
            os.makedirs(result_path[0])
            os.makedirs(result_path[1])
            os.makedirs(test_dir)

        if os.path.exists("test"):
            shutil.rmtree("test")
        os.makedirs("test")
        subprocess.call(["python", dir_ + "main.py",
                         "--dataset_a", data_paths[0],
                         "--dataset_b", data_paths[1],
                         "--which_direction", direction,
                         "--phase", "test",
                         "--checkpoint_dir", checkpoint_dir,
                         "--test_dir", "test",
                         "--test", del_dirs[1],
                         "--test1", del_dirs[0]])
        subprocess.call(["python", dir_ + "main.py",
                         "--dataset_a", del_dir,
                         "--dataset_b", del_dir1,
                         "--which_direction", opposite_direction,
                         "--phase", "test",
                         "--checkpoint_dir", checkpoint_dir,
                         "--test_dir", test_dir,
                         "--test", res0,
                         "--test1", res1])
        shutil.rmtree(del_dir)
        shutil.rmtree(del_dir1)
        if(direction=="BtoA"):
            shutil.rmtree("test")
            shutil.rmtree(test_dir)

    d = "../RankSVM_DATA/Shoes/2/Histogramm/CycleGan/DATA/"
    d1 = "RESULT/"
    for i in range(FLAGS.n_anchor_images):
        for f in range(FLAGS.n_features_embedding):
            GenerateImages2bins([d+str(i) + "_" + str(f) + "_0/", d+ str(i) + "_" + str(f)+ "_1/"], d+"checkpoint_" + str(i) + "_" + str(f) + "_0_1/",
                                [d1+"generated_" + str(i) + "_" + str(f) +"_0/", d1+"generated_" + str(i) + "_" + str(f) + "_1/"], d1+"test/", "AtoB")
            GenerateImages2bins([d + str(i) + "_" + str(f) + "_0/", d + str(i) + "_" + str(f) + "_1/"],
                                d + "checkpoint_" + str(i) + "_" + str(f) + "_0_1/",
                                [d1 + "generated_" + str(i) + "_" + str(f) + "_0/",
                                 d1 + "generated_" + str(i) + "_" + str(f) + "_1/"], d1 + "test/", "BtoA")


if FLAGS.get_features_for_generated_images:

    #recieve true features
    for i in range(FLAGS.n_anchor_images):
        for f in range(FLAGS.n_features_embedding):
            svm_model = "../CycleGAN_shoes/"
            d1 = "RESULT/"
            for b in range(FLAGS.n_bins):
                data_path_ = "generated_" + str(i) + "_" + str(f) + "_" + str(b) + "/"
                GetFeatures(svm_model, d1 + data_path_, d1 + "TrueValue/" + data_path_, FLAGS.n_features_truth)

if FLAGS.get_Myfeatures_for_real_images:
    #recieve true features
    svm_model = "../CycleGAN_shoes/Experiment/My_"
    result_dir = "../CycleGAN_shoes/Experiment/RANDOM/LARGE_" + str(FLAGS.n_features_embedding) + "/"
    data_path = "../CycleGAN_shoes/DATA_ALL_LATENT1/"
    GetFeatures(svm_model, data_path, result_dir, FLAGS.n_features_embedding)



if FLAGS.calculate_colour_histogramm_similarity_for_generated_images:
    data_path = "../CycleGAN_shoes/test_AtoB_1/"
    data_path1 = "../CycleGAN_shoes/test_BtoA_1/"
    data = glob(os.path.join(data_path, "*.jpg"))
    features = []
    features1 = []
    for d in data:
        basename = os.path.basename(d)
        features.append(GetHistogramm(d))
        features1.append(GetHistogramm(data_path1 + basename))
    features = np.array(features)
    features1 = np.array(features1)
    centr = np.mean(features, axis=0)
    centr1 = np.mean(features1, axis=0)
    result = 0.
    for i in range(features.shape[0]):
        res = chi2_distance(features[i], centr, eps=1e-10)
        res1 = chi2_distance(features[i], centr1, eps=1e-10)
        result += float(res < res1)
    print(result / features.shape[0])

    result1 = 0.
    for i in range(features1.shape[0]):
        res = chi2_distance(features1[i], centr, eps=1e-10)
        res1 = chi2_distance(features1[i], centr1, eps=1e-10)
        result1 += float(res < res1)
    print(result1 / features1.shape[0])
    np.savetxt("centresFirst.txt" ,np.array([centr, centr1]))
    #print(chi2_distance(im1, im2, eps=1e-10))

if FLAGS.calculate_colour_histogramm_similarity_for_real_images:
    data_path = "../CycleGAN_shoes/test_AtoB_1/"
    result_dir = ""
    data = glob(os.path.join(data_path, "*.jpg"))
    features = []
    for d in data:
        basename = os.path.basename(d)
        features.append(GetHistogramm(d))
    features = np.array(features)
    cecnters = np.genfromtxt("centresFirst.txt")
    for i in range(features.shape[0]):
        res = chi2_distance(features[i], centr, eps=1e-10)
        res1 = chi2_distance(features[i], centr1, eps=1e-10)
    print(result / features.shape[0])

    result1 = 0.
    for i in range(features1.shape[0]):
        res = chi2_distance(features1[i], centr, eps=1e-10)
        res1 = chi2_distance(features1[i], centr1, eps=1e-10)
        result1 += float(res < res1)
    print(result1 / features1.shape[0])
    np.savetxt("centresFirst.txt" ,np.array([centr, centr1]))

if FLAGS.histogramm_classifier:
    data_path="../RankSVM_DATA/result/"
    data = glob(os.path.join(data_path, "*.jpg"))
    centers = np.genfromtxt("centresFirst.txt")
    random_result = np.random.uniform(0.,1.,len(data))
    result = []
    n_positive = 0.
    for d_i in range(len(data)):
        d = data[d_i]
        features = GetHistogramm(d)
        res = chi2_distance(features, centers[0], eps=1e-10)
        res1 = chi2_distance(features, centers[1], eps=1e-10)
        #print(d, res/res1)
        result.append([res1/res, float(os.path.basename(d).split("_")[0]), random_result[d_i]])
        if res < res1:
            n_positive += 1
    print(n_positive / len(data))
    result.sort(key=lambda  x:x[1])
    zeros = 0.
    binary_quality = 0.
    for r in range(len(result)):
        if (result[r][0] < 1.):
            zeros += 1
        local_quality = (zeros + (n_positive - (r+1-zeros))) / len(data)
        if local_quality > binary_quality:
            binary_quality = local_quality
    print(binary_quality)


    result.sort(key=lambda x: x[0])
    n_bins = 5
    step = (result[-1][0] - result[0][0]) / n_bins
    bin = 0
    for r in result:
        if r[0] >= (bin+1)*step + result[0][0]:
            bin+=1
        r.append(bin)
    result.sort(key=lambda x: x[1])

    bin_index = 3
    pairs_quality = 0.
    bin = [0 for i in range(n_bins)]
    for r in range(len(result)):
        b = 0
        while b < result[r][bin_index]:
            pairs_quality += bin[b]
            b+=1
        bin[result[r][bin_index]] += 1
    n_pairs = 0.
    for i in range(n_bins):
        n_pairs += bin[i] * (len(result) - bin[i])
    n_pairs /= 2.
    print(pairs_quality / n_pairs)

    pairs_quality = 0.
    result.sort(key=lambda x: x[2])
    bin = [0 for i in range(n_bins)]
    for r in range(len(result)):
        b = 0
        while b < result[r][bin_index]:
            pairs_quality += bin[b]
            b+=1
        bin[result[r][bin_index]] += 1
    n_pairs = 0.
    for i in range(n_bins):
        n_pairs += bin[i] * (len(result) - bin[i])
    n_pairs /= 2.
    print(pairs_quality / n_pairs)

if FLAGS.delete_white_border:
    data_path = "../CycleGAN_shoes/DATA_ALL_LATENT/"
    result_dir = "../CycleGAN_shoes/DATA_ALL_LATENT1/"
    SaveImagesWithoutWhiteBorder(data_path, result_dir)




if FLAGS.check_the_mean_distance_generated_images:
    #check the mean distance in True embedding
    def GetFeaturesForAllBins(image_number, feature_number):
        d1 = "RESULT/"
        res = {}
        n_bins =FLAGS.n_bins
        for b in range(FLAGS.n_bins):
            data_path_ = "generated_" + str(image_number) + "_" + str(feature_number) + "_" + str(b) + "/"
            data_path = d1 + "TrueValue/" + data_path_
            data = glob(os.path.join(data_path, "*.jpg"))
            for d in data:
                basename = os.path.basename(d)
                features_value = basename.split("_")[-FLAGS.n_features_truth-1:-1]
                features_value = np.array([float(f_v) for f_v in features_value])
                if (b == 0):
                    res[basename.split("_")[-1]] = [features_value]
                else:
                    res[basename.split("_")[-1]].append(features_value)

        mean_distance = [0. for i in range(n_bins-1)]
        mean_differens = [[] for i in range(n_bins-1)]
        for k in res.keys():
            for b in range(FLAGS.n_bins - 1):
                mean_distance[b] += np.linalg.norm(res[k][b] - res[k][b+1])
                mean_differens[b].append(res[k][b] - res[k][b+1])
        n_examples = len(res.keys())
        for b in range(FLAGS.n_bins - 1):
            mean_distance[b] /= n_examples
            mean_differens[b] = np.array(mean_differens[b])
            print("VARIANCE= ", np.var(mean_differens[b], 0), b)
            mean_differens[b] = np.mean(mean_differens[b], 0)


        print("MEAN DISTANCE = ", mean_distance)
        return mean_differens[0]


    d1 = "RESULT/"
    for image_number in range(FLAGS.n_anchor_images):
        res = []
        for feature_number in range(FLAGS.n_features_embedding):
            res.append(GetFeaturesForAllBins(image_number, feature_number))
        res = np.array(res)
        np.savetxt(d1 + "TrueValue/" + str(image_number) + ".txt", res)














