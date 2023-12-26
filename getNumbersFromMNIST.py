from operator import index
from os import name
import random
import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt

numOfImages = 15
bannedNumIndices = [132]
mndata = MNIST('samples')
images, labels = mndata.load_training()
maxIndex = len(labels)

#This file is only run when you want to check specific images
if(__name__ == "__main__"):
  checkIndex = 132
  print(labels[checkIndex])
  plt.imshow(np.resize(np.array(images[checkIndex]),(28,28)))
  plt.show()
else:
  counter = 0
  images2 = [[]] * numOfImages
  labels2 = []
  if input("Press 1 if you want to work with neighboring images: ") == "1":
    indices = []
    i = maxIndex + 1
    while (i < maxIndex):
      i = int(input("Type the starting index less than {}: ".format(maxIndex)))
    while counter < numOfImages:
      if i in bannedNumIndices:
        i += 1
        continue
      indices.append(i)
      images2[counter] = images[i]
      labels2.append(labels[i])
      counter += 1
      i += 1
  else:
    indices = random.sample(range(1,maxIndex), numOfImages)
    for i in indices:
      images2[counter] = (images[i])
      labels2.append(labels[i])
      counter += 1

def runSamples(index):
  arr = np.resize(np.array(images2[index]),(28,28))
  if input("Press 1 if you want to view the images in command prompt: ") == "1":
    for i in range(len(images2[index])):
      for j in images2[index][i]:
        print(str(j).replace('0','  ').replace('53','. ').replace('178','o ').replace('255','# '),end="")
      print()
  return (arr,labels2[index],indices[index])
