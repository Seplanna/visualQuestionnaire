from FeaturesFromCSV import *
from utils import GetHistogramm
from glob import glob

#take only shoes and boots
def FilterImagesBySubcategory(data_path, dataset_with_metadata, list_of_subcategories, result_path):
    meta_data, n_features, first_line, dictionary = ParceFile(dataset_with_metadata)
    dictionary = dictionary["SubCategory"]
    print (dictionary)
    data = GetData(data_path)
    for d in data:
        basename = os.path.basename(d)
        d_ = basename.split("_")[-1][:-4]
        for sub_cat in list_of_subcategories:
            if meta_data[d_][dictionary[sub_cat][0]] > 0:
                subprocess.call(['cp', d, result_path + os.path.basename(d)])




#take only shoes with high heel or flat
def FilterImageByHeelHigh(data_path, dataset_with_metadata, result_path):
    #os.makedirs(result_path)
    meta_data, n_features, first_line, dictionary = ParceFile(dataset_with_metadata)
    dictionary = dictionary["HeelHeight"]
    print (dictionary.keys())
    data = GetData(data_path)
    for d in data:
        basename = os.path.basename(d)
        d_ = basename.split("_")[-1][:-4]
        for k in dictionary.keys():
            if meta_data[d_][dictionary[k][0]] > 0 and (dictionary[k][1] / 2 == 0 or dictionary[k][1] / 2 == 2):
                subprocess.call(['cp', d, result_path + str((dictionary[k][1] / 2) / 2) + "_" + os.path.basename(d)])


#Take only red or white shoes
def TakeAllImagesWithSomeValues(data_path):
    res = {}
    data = GetData(data_path)
    for d in data:
        d_ = os.path.basename(d).split("_")[:2]
        key = d_[0] + "_" + d_[1]
        if key not in res:
            res[key] = []
        res[key].append(d)
    return res

def AllImagesWithSpecificColour(image_example, data_path, result_path):
    one_bin = TakeAllImagesWithSomeValues(data_path)
    min_len = 1e+10
    for k in one_bin.keys():
        if (min_len > len(one_bin[k])):
            min_len = len(one_bin[k])
    min_len = float(min_len)
    print(min_len)
    n_samples = 0.4 * min_len
    samples_histogramm = GetHistogramm(image_example)

    for k in one_bin.keys():
        index = 0
        distances = []
        for d in one_bin[k]:
            if (index % 100 == 0):
                print(index)
            index += 1
            histogramm = GetHistogramm(d)
            distane = np.linalg.norm(histogramm - samples_histogramm)
            distances.append([distane , d])
        distances.sort(key=lambda x:x[0])
        for i in range(int(n_samples)):
            subprocess.call(["cp", distances[i][1], result_path + os.path.basename(distances[i][1])])

if __name__ == '__main__':
    first_res = "../CycleGAN_shoes/Toy/shoes_boots/"
    second_res = "../CycleGAN_shoes/Toy/shoes_boots_heels/"
    third_res = "../CycleGAN_shoes/Toy/shoes_boots_heels_white_black/"
    res_dirs = [first_res, second_res, third_res]
    for r in res_dirs:
        os.makedirs(r)
    FilterImagesBySubcategory("../CycleGAN_shoes/DATA_ALL_LATENT1/",
                                          "../CycleGAN_shoes/ut-zap50k-data/meta-data-bin.csv",
                                          ["SubCategory.Over.the.Knee",
                                           "SubCategory.Knee.High",
                                           "SubCategory.Mid.Calf",
                                           "SubCategory.Ankle"
                                           ], first_res + "1_")

    FilterImagesBySubcategory("../CycleGAN_shoes/DATA_ALL_LATENT1/",
                                          "../CycleGAN_shoes/ut-zap50k-data/meta-data-bin.csv",
                                          ['SubCategory.Flats', 'SubCategory.Heels'
                                           ], first_res + "0_")

    FilterImageByHeelHigh(first_res, "../CycleGAN_shoes/ut-zap50k-data/meta-data-bin.csv",
                          second_res)

    AllImagesWithSpecificColour("../CycleGAN_shoes/My_5000/2/2.65471533126_4.44508911973_-1.07787978921_-7.34528466874_-0.0498842349829_6.87538099836_8025670.2021.jpg",
                                second_res,
                                third_res + "1_")

    AllImagesWithSpecificColour("../CycleGAN_shoes/My_5000/shoes_boots_heels/2_0_7.52746913204_2.14721116296_-1.51191199544_-2.47253086796_0.244843632019_2.67008852188_8043444.270.jpg",
                                second_res,
                                third_res + "0_")

