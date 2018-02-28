import os
import numpy as np

def LabelsToyDataShoes(d):
    basename = os.path.basename(d).split("_")[:3]
    return np.array([float(f) for f in basename])

def EmbeddingToyDataShoes(d):
    basename = os.path.basename(d).split("_")[3:-1]
    return np.array([float(f) for f in basename])

def Embedding3Features(d):
    basename = os.path.basename(d).split("_")[1:4]
    return np.array([float(f) for f in basename])


def Embedding(EmbbedingName):
    if EmbbedingName == "ToyShoes":
        return EmbeddingToyDataShoes
    if EmbbedingName == "3Features":
        return Embedding3Features
    if EmbbedingName == "LabelToyShoes":
        return LabelsToyDataShoes