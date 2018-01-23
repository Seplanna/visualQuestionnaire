import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import transform
from PIL import Image
import gist

import scipy.io as sio
import scipy.misc
import h5py
import os

def ImgDeleteWhite(img):
  left = 0
  while (np.min(img[:, left]) == 255.):
    left += 1

  right = img.shape[1] - 1
  while (np.min(img[:, right]) == 255.):
    right -= 1

  top = 0
  while (np.min(img[top, :]) == 255.):
    top += 1

  bottom = img.shape[0] - 1
  while (np.min(img[bottom, :]) == 255.):
    bottom -= 1

  return img[top:bottom, left:right, :]

def GetHistogramm(im):
    # grab the image channels, initialize the tuple of colors,
    # the figure and the flattened feature vector
    image = cv2.imread(im)
    chans = cv2.split(image)
    colors = ("b", "g", "r")
    plt.figure()
    plt.title("'Flattened' Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")

    hist = cv2.calcHist([chans[2], chans[1], chans[0]], [0, 1, 2], None,
                        [4, 4, 4], [0, 256, 0, 256, 0, 256])

    features = hist.flatten()
    return features

def chi2_distance(histA, histB, eps=1e-10):
    # compute the chi-squared distance
    d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
                      for (a, b) in zip(histA, histB)])

    # return the chi-squared distance
    return d

def ResizeImage(im):
    imsize = (100, 150)
    pilimg = Image.open(im)
    img = np.asarray(pilimg)
    img = ImgDeleteWhite(img)
    img_resized = transform.resize(img, imsize, preserve_range=True).astype(np.uint8)
    return img_resized

def Gist(im):
    img_resized = ResizeImage(im)
    desc = gist.extract(img_resized)
    return list(desc)

def GetData(data_path):
    data = []
    for root, subdirs, files in os.walk(data_path):
        for f in files:
            #print(f)
            if os.path.splitext(f)[1].lower() in ('.jpg', '.jpeg'):
                data += [os.path.join(root, f)]
    return data

def GetOrderingFromImagePath(data_path):
    mat_contents = sio.loadmat(data_path)
    data = mat_contents['imagepath']
    res = {}
    for i in range(len(data)):
        res[i] = data[i][0][0]
        print (data[i][0][0])
    return res

def GetLabelsFromLexiFile(data_path):
    f = h5py.File(data_path, 'r')
    variables1 = f.items()
    data = [f[element[0]][:] for element in f["mturkOrder"]]
    return data

def GetFeaturesGist(data_path):
    f = h5py.File(data_path, 'r')
    return f['color_feats'], f["gist_feats"]

def inverse_transform(images):
  return (images+1.)/2.

def save_images(images, size, image_path):
  return imsave(inverse_transform(images), size, image_path)

def imsave(images, size, path):
  image = np.squeeze(merge(images, size))
  return scipy.misc.imsave(path, image)

def merge(images, size):
  print(images.shape, size)
  h, w = images.shape[1], images.shape[2]
  if (images.shape[3] in (3,4)):
    c = images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img
  elif images.shape[3]==1:
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
    return img
  else:
    raise ValueError('in merge(images,size) images parameter '
                     'must have dimensions: HxW or HxWx3 or HxWx4')


def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=False, grayscale=False):
  image = imread(image_path, grayscale)
  #image = load_image(image)[1]
  return transform1(image, input_height, input_width,
                   resize_height, resize_width, crop)


def imread(path, grayscale = False):
  if (grayscale):
    return scipy.misc.imread(path, flatten = True).astype(np.float)
  else:
    return scipy.misc.imread(path).astype(np.float)

def transform1(image, input_height, input_width,
              resize_height=64, resize_width=64, crop=True):

  cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
  return np.array(cropped_image)/127.5 - 1.