import os
import random
import mnist
import numpy as np

RAW_IMAGES = mnist.train_images()/255
RAW_LABELS = mnist.train_labels()
TEST_IMAGES = mnist.test_images()/255
TEST_LABELS = mnist.test_labels()
BANNED_INDICES = [132, 41299, 49212]
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def getStartAndEndInd(maxIndex):
  startingIndex = maxIndex
  while (startingIndex > maxIndex-1):
    startingIndex = int(input("Type the starting index less than {}: ".format(maxIndex)))
  endingIndex = maxIndex
  while ((endingIndex > maxIndex-1) or (endingIndex < startingIndex)):
    endingIndex = int(input("Type the ending index more than {} and less than {}: ".format(startingIndex, maxIndex)))
  return (startingIndex, endingIndex)

def getImagesFromMNIST(useTest = False):
  imageData = TEST_IMAGES if useTest else RAW_IMAGES
  labelData = TEST_LABELS if useTest else RAW_LABELS
  bannedIndices = [] if useTest else BANNED_INDICES
  maxIndex = len(imageData) 
  indices = []
  userInp = input("Press 1 if you want to work with neighboring images (this excludes banned images)\nPress 2 to use pre-selected images\nOtherwise sample randomly:\n") 
  if userInp == "1":
    i, endI = getStartAndEndInd(maxIndex)
    images = []
    labels = []
    while i < endI:
      if i in bannedIndices:
        print("index {} is banned".format(i))
        i += 1
        continue
      images.append(imageData[i])
      indices.append(i)
      labels.append(labelData[i])
      i += 1
  elif userInp == "2":
    selectedIndices = [0,1,2]
    images = np.array(imageData)[selectedIndices]
    labels = np.array(labelData)[selectedIndices]
    indices = selectedIndices
  else:
    print("choosing random images")
    numOfImages = int(input("Number of images to use (more than 1 and less than {}): ".format(maxIndex - len(bannedIndices))))
    while(numOfImages<1 or numOfImages > maxIndex-1):
      numOfImages = int(input("Number of images to use (more than 1 less than {}): ".format(maxIndex - len(bannedIndices))))
    images = []
    labels = []
    indices = random.sample(list(filter(lambda x: (x not in bannedIndices), range(1, maxIndex))), numOfImages)
    for i in indices:
      images.append(imageData[i])
      labels.append(labelData[i])
  print("Selected Indices: ", indices)
  return (np.array(images, dtype=np.float32), np.array(labels), np.array(indices))

def getOneImageFromMNIST(index):
  return (np.array(RAW_IMAGES[index]), RAW_LABELS[index])

def geAllMNISTImgs():
  return (np.array(RAW_IMAGES,dtype=np.float32), RAW_LABELS)

def geAllMNISTTestImgs():
  return (np.array(TEST_IMAGES, dtype=np.float32), TEST_LABELS)