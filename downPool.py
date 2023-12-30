
from getNumbers import maxIndex, getOneImageFromMNIST
from showNumbers import showImgsOnPlt
import numpy as np
from skimage.util.shape import view_as_blocks

def downSizeByHalf(img):
  return np.max(np.max(view_as_blocks(img, block_shape=(2,2)), axis=3), axis=2)

if __name__ == '__main__':
  #run this file to test down pooling
  index = maxIndex
  while (index >= maxIndex or index < 0):
    index = int(input("Type image id (max {}): ".format(maxIndex)))
  img, label = getOneImageFromMNIST(index)
  img2 = downSizeByHalf(img)
  img3 = downSizeByHalf(img2)
  showImgsOnPlt([img, img2, img3], [label]*3, [index]*3)
  