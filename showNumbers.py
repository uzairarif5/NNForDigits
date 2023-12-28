
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
  userInp = input("Press 1 to check range of mnist images\nPress 2 to check range of ownDataset images\n")
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