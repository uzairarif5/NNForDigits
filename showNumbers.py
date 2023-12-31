
import matplotlib.pyplot as plt
import numpy as np
from getNumbers import maxIndex, rawImages, rawLabels, getLabelsOfOwnImages, getSpecificOwnImages

def getStartAndEndInd(maxI):
  startingIndex = maxI
  while (startingIndex > maxI-1):
    startingIndex = int(input("Type the starting index less than {}: ".format(maxI)))
  endingIndex = maxI
  if(startingIndex < maxI-10):
    while ((endingIndex > maxI-1) or (endingIndex < startingIndex)):
      endingIndex = int(input("Type the ending index more than {} and less than {}: ".format(startingIndex, maxI)))
  return (startingIndex, endingIndex)

def showImgsOnPlt(images, labels, indices):
    curIndex = 0
    def toggle_images(event):
      nonlocal curIndex
      nonlocal indices
      if(event.key == "right" and curIndex < (len(labels)-1)):
        curIndex += 1
        plt.title("smallIndex: {:>4}, bigIndex: {:>6}, label: {:>3}".format(curIndex, indices[curIndex], labels[curIndex]))
        plt.imshow(images[curIndex])
        plt.draw()
      elif(event.key == "left" and curIndex > 0):
        curIndex -= 1
        plt.title("smallIndex: {:>4}, bigIndex: {:>6}, label: {:>3}".format(curIndex, indices[curIndex], labels[curIndex]))
        plt.imshow(images[curIndex])
        plt.draw()

    print("use left or right key to change picture")
    plt.connect('key_press_event', toggle_images)
    plt.title("smallIndex: {:>4}, bigIndex: {:>6}, label: {:>3}".format(curIndex, indices[curIndex], labels[curIndex]))
    plt.imshow(np.resize(images[0], (28,28)))
    plt.show()

if(__name__ == "__main__"):
  userInp = input("Press 1 to check range of mnist images\nPress 2 to check range of ownDataset images\nPres 3 to use pre-selected indices\n")
  if userInp == "1":
    startingIndex, endingIndex = getStartAndEndInd(maxIndex)
    showImgsOnPlt(rawImages[startingIndex:endingIndex], rawLabels[startingIndex:endingIndex], list(range(startingIndex, endingIndex)))
  elif userInp == "2":
    labels = getLabelsOfOwnImages()
    startingIndex, endingIndex = getStartAndEndInd(len(labels))
    labels = labels[startingIndex: endingIndex]
    print("selected {} images: {}".format(len(labels), labels))
    images = getSpecificOwnImages(startingIndex, endingIndex)
    showImgsOnPlt(images, labels, list(range(startingIndex, endingIndex)))
  elif userInp == "3":
    selectedIndices = [24896,20807,40213,35165,50357,53533,4342,46279,42637,2030,41720,13403,20685,6033,54889,13839,54060,9873,35465,36749,26952,4634,23008,41453,7613,42127,37885,10874,56420,37959,45454,4335,53982,49555,45511,33778,48875,39498,31288,8455,36431,26419,54450,42820,9034,38231,41017,55122,9155,10151,43155,55107,27069,7338,45208,5287,5736,26378,50909,500,49014,27544,50625,21515,8489,45814,21620,32137,17629,41781,39070,35422,31446,8475,9048,4215,43562,45727,17673,1581,27620,56265,6404,19570,32352,39617,30246,12617,4539,58200,3824,3425,39581,26197,21310,9297,7141,28749,43455,44491,20673,5431,3460,33649,53357,6593,3328,2493,22306,31452,26701,52485,19588,18007,17362,2218,144,58339,3493,59480,56921,54377,18222,57898,56377,19103,19924,48807,44848,56528,4216,44161,53061,50687,27972,14883,4508,34054,20068,47419,20642,20080,9561,12616,14991,5488,17435,35060,16407,32953,25867,40241,5574,35343,12026,12660,24218,23289,4615,29882,12877,7297,41831,48028,47901,45738,38112,34432,35997,15060,4889,209,28301,31781,58089,33172,40722,49467,35706,44683,28633,42758,9345,46516,9567,7570,4482,30948,41443,29782,30681,52035,24532,18685,55027,30965,1108,2539,40074,47386,43517,58462,31274,35944,5475,22492,37368,56159,13937,33442,21211,34145,52983,4082,40296,48394,17751,36848,59598,23215,49720,37810,5369,53480,8284,41863,42965,6905,2359,53399,33694,38065,8765,45436,46653,39933,27678,26831,55620,7797,16598,14752,19906,42878,23320,52218,8304,34644,9079,3323,58916,12303,8716,13030,8551,17534,54199,44579,30793,40618,19582,2275,35083,30,4258,30967,52696,56530,8666,21199,23602,1778,47493,7325,4002,25034,9749,25547,6016,49268,53502,51763,26178,359,44045,11341,19335,40400,18487,43554,42183,29751,53691,38488,56716,38982,27030,22063,19280,47539,49142,53789,2098,12313,54707,16044,53765,38859,2627,6757]
    showImgsOnPlt(np.array(rawImages)[selectedIndices], np.array(rawLabels)[selectedIndices], selectedIndices)
