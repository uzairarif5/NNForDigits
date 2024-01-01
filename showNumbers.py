
import matplotlib.pyplot as plt
import numpy as np
from getNumbers import MAX_INDEX, getImagesFromMNIST, getLabelsOfOwnImages, getSpecificOwnImages

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
    images, labels, indices = getImagesFromMNIST()
    showImgsOnPlt(images, labels, indices)
