import numpy as np
import Tkinter as tk
from PIL import Image, ImageTk
import matplotlib
import progressbar
matplotlib.use('TkAgg')

from QuestionsInterpretability import Real1_one_feature_from_given_features
from Evaluation import *
from Embedding import *
from utils import GetData
#----------------------------
#1.Take all images that satisfy preference so far
#2.Devide them by bins
#3.Resieve question
#4.Take images from one bin
#----------------------------
class ImageSearch(object):
    def __init__(self, data_path, n_bins, n_pictures_per_bin, n_features, Embedding):
        # download dataset------------------------------------------
        self.data = glob(os.path.join(data_path, "*.jpg"))
        self.n_bins = n_bins
        self.n_features = n_features
        self.features = []
        for d in self.data:
            self.features.append(Embedding(d))
        self.features = np.array(self.features)
        self.n_pictures_per_bin = n_pictures_per_bin
        self.feature_n = 0
        # --------------------------

        self.step = 0
        self.n_pictures = self.n_bins * self.n_pictures_per_bin
        self.show_images()

    def show_images(self):
        images_name, self.data_by_bins = Real1_one_feature_from_given_features(self.features,
                                                                     self.data,
                                                                     self.n_bins,
                                                                     self.n_pictures_per_bin,
                                                                     self.feature_n % self.n_features)

        self.images_name = []
        for i in images_name:
            self.images_name += i

    def calculate(self, answer):
        self.answer = int(answer)
        self.feature_n += 1
        self.features = self.features[self.data_by_bins[answer]]
        self.data = [self.data[d] for d in self.data_by_bins[answer]]
        self.step += 1
        self.show_images()

class UserSimuliator(object):
    def __init__(self, data_path, n_bins, n_pictures_per_bin, n_features, Embedding, data_path_truth, Embedding_truth, im):
        self.data_path = data_path
        self.Embedding = Embedding
        self.image_search = ImageSearch(data_path, n_bins, n_pictures_per_bin, n_features, Embedding)
        data_truth_ = GetData(data_path_truth)
        self.data_truth = {}
        for d in data_truth_:
            self.data_truth[os.path.basename(d).split("_")[-1]] = Embedding_truth(d)
        self.chosen_picture = im
        self.truth_value = self.data_truth[os.path.basename(self.chosen_picture).split("_")[-1]]
        self.image_search.show_images()

    def answer(self):
        mean_truth_vector = []
        for b in range(self.image_search.n_bins):
            bin_vector = np.zeros(self.image_search.n_features)
            for im in range(self.image_search.n_pictures_per_bin):
                bin_vector += \
                self.data_truth[os.path.basename(
                    self.image_search.images_name[b*self.image_search.n_pictures_per_bin + im]).split("_")[-1]]
            bin_vector /= self.image_search.n_pictures_per_bin
            mean_truth_vector.append(bin_vector)
        mean_truth_vector = np.array(mean_truth_vector)
        mean_truth_vector -= self.truth_value
        return np.argmin(np.linalg.norm(mean_truth_vector, axis=1))

    def min_distance_to_bin(self, bin):
        distances = []
        for im in self.image_search.data_by_bins[bin]:
            distances.append(self.data_truth[os.path.basename(
                    self.image_search.data[im]).split("_")[-1]])
        distances = np.array(distances)
        distances -= self.truth_value
        return np.sort(np.linalg.norm(distances, axis=1))

    def put_chosen_picture(self, im):
        self.chosen_picture = im
        self.truth_value = self.data_truth[os.path.basename(self.chosen_picture).split("_")[-1]]
        self.image_search = ImageSearch(self.data_path,
                                        self.image_search.n_bins,
                                        self.image_search.n_pictures_per_bin,
                                        self.image_search.n_features,
                                        self.Embedding)

    def one_iteration(self):
        results = []
        for bin in range(self.image_search.n_bins):
            res = self.min_distance_to_bin(bin)
            result = np.zeros(4)
            b = 0
            last_res = 0
            for r in res:
                if np.abs(r - last_res) > 1e-10:
                    b += 1
                    last_res = r
                result[b] += 1
            results.append(result)
        answer = self.answer()
        self.image_search.calculate(answer)
        return results[answer] / np.sum(results[answer])
#----------------------------
class ShowImages(tk.Frame):

    def __init__(self, parent, data_path, n_bins, n_pictures_per_bin, n_features, Embedding):
        tk.Frame.__init__(self, parent)
        #download dataset------------------------------------------
        self.image_search = ImageSearch(data_path, n_bins, n_pictures_per_bin, n_features, Embedding)

        # -------------------------------
        #Get random Example
        self.chosen_picture = random.choice(self.image_search.data)
        print(self.chosen_picture)
        #---------------------------------------------------------
        self.get_images_from_names()

        self.submits = []

        self.image_width = 1.5*self.images[0].width()
        self.image_high = 1.5*self.images[0].height()
        image_width = self.image_width
        image_high = self.image_high

        for i in range(self.image_search.n_pictures):
            self.submits.append(tk.Button(self, command=lambda i=i: self.calculate(i)))
            if (i < len(self.images)):
                self.submits[-1].config(image=self.images[i])
            else:
                self.submits[-1].config(image=self.images[-1])
            self.submits[-1].place(x= i%self.image_search.n_pictures_per_bin * image_width,
                                   y= (i/self.image_search.n_pictures_per_bin + 1) * (image_high + 10), width=image_width,
                                   height=image_high)

        self.submits.append(tk.Button(self))
        self.chosen_picture_image = ImageTk.PhotoImage(Image.open(self.chosen_picture))
        self.submits[-1].config(image=self.chosen_picture_image)
        self.submits[-1].place(x=0,y=0,width=image_width, height=image_high)

        self.submits.append(tk.Button(self))
        self.skip = ImageTk.PhotoImage(Image.open("del.png"))
        self.submits[-1].config(image=self.skip, command=lambda i=i: self.calculate(-1))
        self.submits[-1].place(x=2*image_width,y=0,width=image_width, height=image_high)

        self.constraint_images = []
        for i in range(self.image_search.n_bins):
            self.submits.append(tk.Button(self))
            self.constraint_images.append(ImageTk.PhotoImage(Image.open("del.png")))
            self.submits[-1].config(image=self.constraint_images[-1], command=lambda i=i: self.calculate(i))
            self.submits[-1].place(x=self.image_search.n_pictures_per_bin*image_width,
                                   y=(i+1)*(image_high + 10),width=image_width, height=image_high)

        self.entry = tk.Entry(self)
        self.output = tk.Label(self, text="")



    def show_images(self):
        self.image_search.show_images()

    def get_images_from_names(self):
        self.images = []
        for i in self.image_search.images_name:
            self.images.append(ImageTk.PhotoImage(Image.open(i)))

        root.geometry('%dx%d'     % ((self.image_search.n_pictures_per_bin + 2) * 1.5 * self.images[0].width() + 90,
       (self.image_search.n_bins + 1) * 1.5 * self.images[0].height() + 80))

    def submit_images(self):
        for i in range(len(self.images)):
            self.submits[i].config(image=self.images[i])
        for i in range(self.image_search.n_bins):
            self.constraint_images.append(ImageTk.PhotoImage(Image.open(str(i) + ".jpg")))
            self.submits[i-self.n_bins].config(image=self.constraint_images[-1])
        self.skip = ImageTk.PhotoImage(Image.open(self.data_path[self.anchor_im]))
        self.submits[-self.image_search.n_bins-1].config(image=self.skip)

    def calculate(self, answer):
        self.image_search.calculate(answer)
        self.get_images_from_names()
        self.submit_images()




"""if __name__ == "__main__":
    root = tk.Tk()
    root.geometry('+%d+%d' % (100, 100))

    ShowImages(root, "../CycleGAN_shoes/Toy/PCA_GIST/0/", 2, 5, 3,
               Embedding("3Features")).pack(fill="both", expand=True)
    #ShowImages(root, "../RankSVM_DATA/Shoes/AutoEncoder/result9/","./RankSVM_DATA/Shoes/Supervised/result/", "result/",
    #           "../RankSVM_DATA/Shoes/AutoEncoder/", 3).pack(fill="both", expand=True)
    root.mainloop()"""

if __name__ == "__main__":
    n_iterations = 7
    n_images = 1000
    data = GetData("../CycleGAN_shoes/Toy/shoes_boots_heels_white_black/")

    data_path0 = "../CycleGAN_shoes/Toy/My_interpretability/0/"
    data_path1 = "../CycleGAN_shoes/Toy/AutoEncoderByClusterDirection/0/"
    data_path2 = "../CycleGAN_shoes/Toy/PCA_AutoEncoder/0/"
    data_path3 = "../CycleGAN_shoes/Toy/PCA_GIST/0/"

    bar = progressbar.ProgressBar(maxval=n_images, \
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

    data_paths = [data_path0, data_path1, data_path2, data_path3]
    simuliators = []
    for method in range(len(data_paths)):
        simuliators.append(UserSimuliator (data_paths[method], 2, 5, 3,
               Embedding("3Features"),  "../CycleGAN_shoes/Toy/shoes_boots_heels_white_black/", Embedding("LabelToyShoes"),
                 random.choice(data)))
    results = np.zeros([len(data_paths), n_iterations, 4])


    bar.start()
    for im in range(n_images):
        bar.update(im + 1)
        im_ = random.choice(data)
        for simuliator in range(len(simuliators)):
            simuliators[simuliator].put_chosen_picture(im_)
            for it in range(n_iterations):
                results[simuliator][it] += simuliators[simuliator].one_iteration()
    bar.finish()

    for simuliator in range(len(simuliators)):
        print ("METHOD = ", simuliator)
        print(results[simuliator]/ n_images)



