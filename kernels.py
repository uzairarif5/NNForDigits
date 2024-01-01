
from getNumbers import getImagesFromMNIST
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


def convColumn(rowNum, k, img):
  r = rowNum[0]
  if(r ==0 or r == len(img)-1):
    return np.zeros(len(img))
  return (np.convolve(img[r-1], k[0]) + np.convolve(img[r], k[1]) + np.convolve(img[r+1], k[2]))[1:-1]

def applyOneKernel(k, img):
  rowNums = np.arange(len(img)).reshape((len(img),1))
  mulRVec = np.vectorize(convColumn, signature="(r), (i, j), (n, m) -> (x)")
  return mulRVec(rowNums, k, img)

def applyKernels(img):
  applyOneKerVec = np.vectorize(applyOneKernel, signature="(k1, k2), (n, m) -> (i, j)")
  mainOutputArr = applyOneKerVec(kernels, img)
  return mainOutputArr

if __name__ == '__main__':
  #run this file to test kernels
  imgs, labels, indices, = getImagesFromMNIST()
  applyKernelVec = np.vectorize(applyKernels, signature="(m, n) -> (i, j, k)")
  filterImages = applyKernelVec(imgs).reshape(4 * len(imgs), 28, 28)
  showImgsOnPlt(filterImages, np.repeat(labels, 4), np.repeat(indices, 4))
  