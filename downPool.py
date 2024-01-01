
from getNumbers import getImagesFromMNIST
from showNumbers import showImgsOnPlt
import numpy as np
from skimage.util.shape import view_as_blocks

def downSizeByHalf(img):
  return np.max(np.max(view_as_blocks(img, block_shape=(2,2)), axis=3), axis=2)

if __name__ == '__main__':
  #run this file to test down pooling
  imgs, labels, indices = getImagesFromMNIST()
  downSizeVectorized = np.vectorize(downSizeByHalf, signature="(m, n) -> (i, j)")
  smallImages = downSizeVectorized(imgs)
  downSizeVectorized = np.vectorize(downSizeByHalf, signature="(m, n) -> (i, j)")
  smallImages2 = downSizeVectorized(smallImages)
  showImgsOnPlt(smallImages2, labels, indices)
  