
from getNumbers import maxIndex, getOneImageFromMNIST
import time
from showNumbers import showImgsOnPlt
import numpy as np

kernels = np.array([
  [
    [-1/4, -2/4, -1/4],
    [0, 0, 0],
    [1/4, 2/4, 1/4]
  ],
  [
    [1/4, 2/4, 1/4],
    [0, 0, 0],
    [-1/4, -2/4, -1/4]
  ],
  [
    [1/4, 0, -1/4],
    [2/4, 0, -2/4],
    [1/4, 0, -1/4]
  ],
  [
    [-1/4, 0, 1/4],
    [-2/4, 0, 2/4],
    [-1/4, 0, 1/4]
  ],
])

def applyOneKernel(k, img):
  curOutputArr = np.zeros((len(img),len(img[0])))
  for r in range(1, len(curOutputArr)-1):
    for c in range(1, len(curOutputArr[0])-1):
      curOutputArr[r][c] += np.sum(img[r-1:r+2, c-1:c+2] * k)
  return curOutputArr

def applyKernels(img):
  applyOneKerVec = np.vectorize(applyOneKernel, signature="(k1, k2), (n, m) -> (i, j)")
  mainOutputArr = applyOneKerVec(kernels, img)
  return mainOutputArr

if __name__ == '__main__':
  #run this file to test kernels
  index = maxIndex
  while (index >= maxIndex or index < 0):
    index = int(input("Type image id (max {}): ".format(maxIndex)))
  img, label = getOneImageFromMNIST(index)
  showImgsOnPlt(applyKernels(img), [label]*(len(kernels)), [index]*(len(kernels)))
  