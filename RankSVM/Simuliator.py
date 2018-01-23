import numpy as np
import Tkinter as tk
from PIL import Image, ImageTk
import random
import os
from glob import glob
import math

from Evaluation import *



class ShowImages(tk.Frame):

    def __init__(self, parent, data_path_bin, data_path_truth, data_path_questions, data_path_histogramm, n_bins):
        tk.Frame.__init__(self, parent)


        del_ = ["Test/2_2_2_2_2_2.05656247919_0.272136914039_-1.51937059729_3.78667569721_-0.620247817651_7980011.16740.jpg",
                "Test/1_0_1_2_2_1.10807503352_0.249567690135_ - 2.3807081615_0.742837890207_- 1.23791180893_8026541.371867.jpg",
                "0_1_2_0_1_0.675851239048_0.157134270511_-1.51695127277_2.19207010031_-1.87046167021_7629043.14.jpg"
                ]
        #download dataset------------------------------------------
        self.n_features = 5
        self.n_features_truth = 10
        self.data_ = glob(os.path.join(data_path_bin, "*.jpg"))
        self.test = glob(os.path.join("../RankSVM_DATA/Shoes/2/Histogramm/result/", "*.jpg"))
        self.data_bin = {}
        self.data_path = {}
        for i in self.data_:
            basename = os.path.basename(i)
            self.data_bin[basename.split("_")[-1]] = np.array([float(f) for f in basename.split("_")[:2 * self.n_features]])
            self.data_path[basename.split("_")[-1]] = i

        data1_ = glob(os.path.join(data_path_truth, "*.jpg"))
        self.data_truth = {}
        for i in data1_:
            basename = os.path.basename(i)
            self.data_truth[basename.split("_")[-1]] = np.array([float(f) for f in basename.split("_")[:self.n_features_truth]])
        self.data_path_questions = data_path_questions
        self.data_path_histogramm = data_path_histogramm
        self.n_bins = n_bins
        #---------------------------------------------------------
        #Get random Example
        self.chosen_picture = random.choice(self.test)
        #self.chosen_picture = del_[1]
        self.chosen_picture = os.path.basename(self.chosen_picture).split("_")[-1]
        print(self.data_bin[self.chosen_picture])
        #---------------------------------------------------------
        self.feature_n = 0
        self.step = 0

        self.n_pictures_per_bin = 3
        self.n_pictures = self.n_bins * self.n_pictures_per_bin
        self.show_images()
        self.get_images_from_names()
        self.estimation = np.zeros(self.n_features)
        self.constraints = {}
        self.Ranking = True

        self.submits = []
        print(len(self.images), self.images_name)
        self.image_width = 1.5*self.images[0].width()
        self.image_high = 1.5*self.images[0].height()
        image_width = self.image_width
        image_high = self.image_high

        for i in range(self.n_pictures):
            self.submits.append(tk.Button(self, command=lambda i=i: self.calculate(i)))
            if (i < len(self.images)):
                self.submits[-1].config(image=self.images[i])
            else:
                self.submits[-1].config(image=self.images[-1])
            self.submits[-1].place(x= i%self.n_pictures_per_bin * image_width,
                                   y= (i/self.n_pictures_per_bin + 1) * (image_high + 10), width=image_width,
                                   height=image_high)

        self.submits.append(tk.Button(self))
        self.chosen_picture_image = ImageTk.PhotoImage(Image.open(self.data_path[self.chosen_picture]))
        self.submits[-1].config(image=self.chosen_picture_image)
        self.submits[-1].place(x=0,y=0,width=image_width, height=image_high)

        self.submits.append(tk.Button(self))
        self.skip = ImageTk.PhotoImage(Image.open("6.jpg"))
        self.submits[-1].config(image=self.skip, command=lambda i=i: self.calculate(-1))
        self.submits[-1].place(x=2*image_width,y=0,width=image_width, height=image_high)

        self.constraint_images = []
        for i in range(self.n_bins):
            self.submits.append(tk.Button(self))
            self.constraint_images.append(ImageTk.PhotoImage(Image.open("6.jpg")))
            self.submits[-1].config(image=self.constraint_images[-1], command=lambda i=i: self.calculate(-2-i))
            self.submits[-1].place(x=self.n_pictures_per_bin*image_width,y=(i+1)*(image_high + 10),width=image_width, height=image_high)

        self.features = []


        self.answers_statistocks = np.zeros([self.n_features, self.n_bins])

        self.entry = tk.Entry(self)
        self.output = tk.Label(self, text="")



    def show_images(self):
        if (self.step < 0):
            self.images_name = GetRandomPicture(self.data_, self.n_pictures)#GetAnchorPictures("Anchor_im")#GetRandomPicture(self.data_, self.n_pictures)
        elif (self.step < 1):
            self.images_name = GetAnchorPictures("Anchor_im1/")[:self.n_pictures]
            #self.images_name[0] = self.chosen_picture


        elif(self.feature_n% (self.n_features) == 0 and not self.Ranking):
            print("RANKING")
            print(self.constraints)
            print(self.data_bin[self.anchor_im][:self.n_features])
            print(self.data_bin[self.chosen_picture][:self.n_features])
            self.images_name, img_ind = GetRanking(self.data_bin,
                                          self.data_bin[self.chosen_picture][self.n_features:],
                                          self.chosen_picture,
                                          self.n_features,
                                          self.n_pictures,
                                          self.constraints)
            print(self.answers_statistocks, self.data_bin[self.chosen_picture][:self.n_features])
            print(self.features)
            self.Ranking = True
            return
        else:
            print("TRUTH = ,", self.data_bin[self.chosen_picture][self.n_features - 1 - self.feature_n% self.n_features])
            print(self.data_bin[self.chosen_picture][:self.n_features])
            print(self.data_bin[self.anchor_im][:self.n_features])

            self.Ranking = False
            self.images_name = []
            images_name1 = []
            images_name1 += GetQuestion(self.data_bin[self.anchor_im], self.data_bin, self.n_bins,
                                           self.feature_n % self.n_features, self.data_path_histogramm, False,
                                            1./(math.sqrt(self.step)), self.n_pictures_per_bin, self.n_features, self.constraints)

            #images_name = []
            #images_name += GetQuestionWithMaximumValue(self.data_bin[self.anchor_im], self.data_bin, self.n_bins,
            #                               self.feature_n % self.n_features, self.data_path_histogramm, False,
            #                                1./(math.sqrt(self.step)), self.n_pictures_per_bin /2, self.n_features, self.constraints)
            #images_name1 = []
            #images_name1 += GetQuestionGenerated(self.data_bin[self.anchor_im], self.data_bin, self.n_bins,
            #                                                               self.feature_n % self.n_features, self.data_path_histogramm, False,
            #                                                               1./(math.sqrt(self.step)), self.n_pictures_per_bin / 2, self.n_features, self.constraints)
            #for b in range(self.n_bins):
                #self.images_name += images_name[b*(self.n_pictures_per_bin / 2): (b+1)*(self.n_pictures_per_bin / 2)]
                #self.images_name += images_name1[b * (self.n_pictures_per_bin / 2): (b + 1) * (self.n_pictures_per_bin / 2)]
            self.images_name = images_name1

            #print(self.data_bin[self.anchor_im])
            #for im in self.images_name:
            #    print(self.data_bin[im])



            res = GetQuestionSave(self.data_bin[self.anchor_im], self.data_path, self.data_bin, self.n_bins,
                                           self.feature_n % self.n_features, self.data_path_histogramm, False,
                                            1./(math.sqrt(self.step)), self.n_features)

            self.feature_n += 1




    def get_images_from_names(self):
        self.images = []
        for i in self.images_name:
            self.images.append(ImageTk.PhotoImage(Image.open(self.data_path[i])))

        root.geometry('%dx%d' % ((self.n_pictures_per_bin+2) * 1.5 * self.images[0].width() + 90,
                                 (self.n_bins + 1)* 1.5 * self.images[0].height() + 80))

    def get_images_from_namesGenerated(self):
        self.images = []
        for i in self.images_name:
            self.images.append(ImageTk.PhotoImage(Image.open(i)))

        root.geometry('%dx%d' % ((self.n_pictures_per_bin+2) * 1.5 * self.images[0].width() + 90,
                                 (self.n_bins + 1)* 1.5 * self.images[0].height() + 80))

    def GetConstraintImages(self):
        feature_n = str(self.n_features-(self.feature_n-1)%self.n_features)
        data = random.choice(glob(os.path.join("../CycleGAN_shoes/test_AtoB_" + feature_n + "/", "*.jpg")))
        data1 = glob(os.path.join("../CycleGAN_shoes/test_BtoA_" + feature_n + "/", "*.jpg"))
        basename = os.path.basename(data).split("_")[-1]
        for d in data1:
            if(os.path.basename(d).split("_")[-1] == basename):
                data1=d
                break
        print("DATA = ", data, data1)
        self.constraint_images = []
        self.constraint_images.append(ImageTk.PhotoImage(Image.open(data)))
        self.constraint_images.append(ImageTk.PhotoImage(Image.open(data1)))

    def submit_images(self):
        for i in range(len(self.images)):
            self.submits[i].config(image=self.images[i])
        #self.constraint_images = []
        #self.GetConstraintImages()
        for i in range(self.n_bins):
            self.constraint_images.append(ImageTk.PhotoImage(Image.open(str(i) + ".jpg")))
            self.submits[i-self.n_bins].config(image=self.constraint_images[-1])
        self.skip = ImageTk.PhotoImage(Image.open(self.data_path[self.anchor_im]))
        self.submits[-self.n_bins-1].config(image=self.skip)

    def calculate(self, answer):
        print("FEATURE_N, ", self.feature_n, self.feature_n%self.n_features)
        print("ANSWER = ", answer)
        print("LEN SUBMITS = ", len(self.submits))
        print("STEP = ", self.step)
        self.answer = int(answer)
        if (answer == -1):
            self.step += 1
            self.show_images()
            self.get_images_from_names()
            self.submit_images()
            print("DISTANCE = ", np.linalg.norm(self.data_truth[self.chosen_picture]- self.data_truth[self.anchor_im]))
            return
        elif (answer < 0):
            self.constraints[self.n_features - 1 - (self.feature_n-1)%self.n_features] = -(answer+2)
            print(self.feature_n-1, self.constraints)
            return

        self.anchor_im = self.images_name[answer]
        for i in range(self.n_features):
            self.answers_statistocks[i][self.data_bin[self.anchor_im][i]] += 1
        self.estimation += self.data_bin[self.anchor_im][self.n_features:]
        self.step += 1
        self.show_images()
        self.get_images_from_names()
        self.submit_images()
        print("DISTANCE = ", np.linalg.norm(self.data_truth[self.chosen_picture] - self.data_truth[self.anchor_im]))







if __name__ == "__main__":
    root = tk.Tk()
    root.geometry('+%d+%d' % (100, 100))
    ShowImages(root, "../RankSVM_DATA/Shoes/Histogramm/result4/", "../RankSVM_DATA/Shoes/Supervised/result/", "result/",
               "../RankSVM_DATA/Shoes/Histogramm/", 3).pack(fill="both", expand=True)
    #ShowImages(root, "../RankSVM_DATA/Shoes/AutoEncoder/result9/","./RankSVM_DATA/Shoes/Supervised/result/", "result/",
    #           "../RankSVM_DATA/Shoes/AutoEncoder/", 3).pack(fill="both", expand=True)
    root.mainloop()
