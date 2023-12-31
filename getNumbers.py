import os
import random
import mnist
import pickle
import numpy as np

bannedNumIndices = [132, 41299, 49212]
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
  userInp = input("Press 1 if you want to work with neighboring images (this excludes banned images)\nPress 2 to use pre-selected images\nOtherwise sample randomly:\n") 
  if userInp == "1":
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
  elif userInp == "2":
    selectedIndices = [24896,20807,40213,35165,50357,53533,4342,46279,42637,2030,41720,13403,20685,6033,54889,13839,54060,9873,35465,36749,26952,4634,23008,41453,7613,42127,37885,10874,56420,37959,45454,4335,53982,49555,45511,33778,48875,39498,31288,8455,36431,26419,54450,42820,9034,38231,41017,55122,9155,10151,43155,55107,27069,7338,45208,5287,5736,26378,50909,500,49014,27544,50625,21515,8489,45814,21620,32137,17629,41781,39070,35422,31446,8475,9048,4215,43562,45727,17673,1581,27620,56265,6404,19570,32352,39617,30246,12617,4539,58200,3824,3425,39581,26197,21310,9297,7141,28749,43455,44491,20673,5431,3460,33649,53357,6593,3328,2493,22306,31452,26701,52485,19588,18007,17362,2218,144,58339,3493,59480,56921,54377,18222,57898,56377,19103,19924,48807,44848,56528,4216,44161,53061,50687,27972,14883,4508,34054,20068,47419,20642,20080,9561,12616,14991,5488,17435,35060,16407,32953,25867,40241,5574,35343,12026,12660,24218,23289,4615,29882,12877,7297,41831,48028,47901,45738,38112,34432,35997,15060,4889,209,28301,31781,58089,33172,40722,49467,35706,44683,28633,42758,9345,46516,9567,7570,4482,30948,41443,29782,30681,52035,24532,18685,55027,30965,1108,2539,40074,47386,43517,58462,31274,35944,5475,22492,37368,56159,13937,33442,21211,34145,52983,4082,40296,48394,17751,36848,59598,23215,49720,37810,5369,53480,8284,41863,42965,6905,2359,53399,33694,38065,8765,45436,46653,39933,27678,26831,55620,7797,16598,14752,19906,42878,23320,52218,8304,34644,9079,3323,58916,12303,8716,13030,8551,17534,54199,44579,30793,40618,19582,2275,35083,30,4258,30967,52696,56530,8666,21199,23602,1778,47493,7325,4002,25034,9749,25547,6016,49268,53502,51763,26178,359,44045,11341,19335,40400,18487,43554,42183,29751,53691,38488,56716,38982,27030,22063,19280,47539,49142,53789,2098,12313,54707,16044,53765,38859,2627,6757]
    images = np.array(rawImages)[selectedIndices]
    labels = np.array(rawLabels)[selectedIndices]
    indices = selectedIndices
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
  return (np.array(images, dtype=np.uint16), np.array(labels), np.array(indices))

def getOneImageFromMNIST(index):
  return (np.array(rawImages[index]), rawLabels[index])

def getLabelsOfOwnImages():
    file = open(dir_path + "/ownDatasetStuff/ownLabels.txt","r")
    labels = list(map(int,(file.read()).split("\n")))
    file.close()
    print("your dataset contains {} images".format(len(labels)))
    return np.array(labels)

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
