import os
import numpy as np

def EmbeddingToyDataShoes(d):
    basename = os.path.basename(d).split("_")[3:-1]
    return np.array([float(f) for f in basename])