import os
import random
import mnist
import numpy as np

RAW_IMAGES = mnist.train_images()
RAW_LABELS = mnist.train_labels()
TEST_IMAGES = mnist.test_images()
TEST_LABELS = mnist.test_labels()
MAX_INDEX = len(RAW_LABELS)
BANNED_INDICES = [132, 41299, 49212]
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def getStartAndEndInd():
  startingIndex = MAX_INDEX
  while (startingIndex > MAX_INDEX-1):
    startingIndex = int(input("Type the starting index less than {}: ".format(MAX_INDEX)))
  endingIndex = MAX_INDEX
  while ((endingIndex > MAX_INDEX-1) or (endingIndex < startingIndex)):
    endingIndex = int(input("Type the ending index more than {} and less than {}: ".format(startingIndex, MAX_INDEX)))
  return (startingIndex, endingIndex)

def getImagesFromMNIST():
  indices = []
  userInp = input("Press 1 if you want to work with neighboring images (this excludes banned images)\nPress 2 to use pre-selected images\nOtherwise sample randomly:\n") 
  if userInp == "1":
    i, endI = getStartAndEndInd()
    images = []
    labels = []
    while i < endI:
      if i in BANNED_INDICES:
        print("index {} is banned".format(i))
        i += 1
        continue
      images.append(RAW_IMAGES[i])
      indices.append(i)
      labels.append(RAW_LABELS[i])
      i += 1
  elif userInp == "2":
    selectedIndices = [0,1,2]
    images = np.array(RAW_IMAGES)[selectedIndices]
    labels = np.array(RAW_LABELS)[selectedIndices]
    indices = selectedIndices
  else:
    print("choosing random images")
    numOfImages = int(input("Number of images to use (more than 1 and less than {}): ".format(MAX_INDEX)))
    while(numOfImages<1 or numOfImages>MAX_INDEX-1):
      numOfImages = int(input("Number of images to use (more than 1 less than {}): ".format(MAX_INDEX)))
    images = []
    labels = []
    indices = random.sample(list(filter(lambda x: (x not in BANNED_INDICES), range(1,MAX_INDEX))), numOfImages)
    for i in indices:
      images.append(RAW_IMAGES[i])
      labels.append(RAW_LABELS[i])
  print("Selected Indices: ", indices)
  return (np.array(images, dtype=np.float32), np.array(labels), np.array(indices))

def getOneImageFromMNIST(index):
  return (np.array(RAW_IMAGES[index]), RAW_LABELS[index])

def geAllMNISTImgs():
  return (np.array(RAW_IMAGES,dtype=np.float32), RAW_LABELS)

def geAllMNISTTestImgs():
  return (np.array(TEST_IMAGES, dtype=np.float32), TEST_LABELS)