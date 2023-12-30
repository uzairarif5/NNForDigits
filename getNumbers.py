import os
import random
import mnist
import pickle
import numpy as np

bannedNumIndices = [132, 41299]
rawImages = mnist.train_images()
rawLabels = mnist.train_labels()
maxIndex = len(rawLabels)
dir_path = os.path.dirname(os.path.realpath(__file__))

def getStartAndEndInd():
  global maxIndex
  startingIndex = maxIndex
  while (startingIndex > maxIndex-1):
    startingIndex = int(input("Type the starting index less than {}: ".format(maxIndex)))
  endingIndex = maxIndex
  while ((endingIndex > maxIndex-1) or (endingIndex < startingIndex)):
    endingIndex = int(input("Type the ending index more than {} and less than {}: ".format(startingIndex, maxIndex)))
  return (startingIndex, endingIndex)

def getImagesFromMNIST():
  indices = []
  if input("Press 1 if you want to work with neighboring images (this excludes banned images), otherwise sample randomly: ") == "1":
    i, endI = getStartAndEndInd()
    images = []
    labels = []
    while i < endI:
      if i in bannedNumIndices:
        print("index {} is banned".format(i))
        i += 1
        continue
      images.append(rawImages[i])
      indices.append(i)
      labels.append(rawLabels[i])
      i += 1
  else:
    print("choosing random images")
    numOfImages = int(input("Choose the number to use (more than 1 and less than {}): ".format(maxIndex)))
    while(numOfImages<1 or numOfImages>maxIndex-1):
      numOfImages = int(input("Choose the number to use (more than 1 less than {}): ".format(maxIndex)))
    images = []
    labels = []
    indices = random.sample(list(filter(lambda x: (x not in bannedNumIndices), range(1,maxIndex))), numOfImages)
    for i in indices:
      images.append(rawImages[i])
      labels.append(rawLabels[i])
  print("Selected Indices: ", indices)
  return (np.array(images), labels, indices)

def getOneImageFromMNIST(index):
  return (np.array(rawImages[index]), rawLabels[index])

def getLabelsOfOwnImages():
    file = open(dir_path + "/ownDatasetStuff/ownLabels.txt","r")
    labels = list(map(int,(file.read()).split("\n")))
    file.close()
    print("your dataset contains {} images".format(len(labels)))
    return labels

def getOwnImages():
    file = open(dir_path + "/ownDatasetStuff/ownImages.dat","rb")
    images = (np.array(pickle.load(file)) * 255).astype("int16")
    file.close()
    return images

def getSpecificOwnImages(startingInd, endingIndex):
    file = open("ownDatasetStuff/ownImages.dat","rb")
    images = (np.array(pickle.load(file))[startingInd: endingIndex] * 255).astype("int16")
    file.close()
    return images
