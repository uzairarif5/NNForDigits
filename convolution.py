import getNumbers
from showNumbers import showImgsOnPlt
from kernels import applyKernels
import numpy as np
from downPool import downSizeByHalf

def filterReluDownpool(inputImage):
  #filter
  applyKernelVec = np.vectorize(applyKernels, signature="(m, n) -> (i, j, k)")
  filterImages = applyKernelVec(inputImage)
  #Relu
  filterImages = np.clip(filterImages,0,255)
  #downPool
  downSizeVectorized = np.vectorize(downSizeByHalf, signature="(m, n) -> (i, j)")
  smallImages = downSizeVectorized(filterImages)
  #output
  return smallImages


def doubleConv(inputImages):
  print("Starting convolution...")
  filterReluDpVec = np.vectorize(filterReluDownpool, signature="(m, n) -> (i, j, k)")
  smallImages1 = filterReluDpVec(inputImages)
  print("First convolution done")
  smallImages2 = (filterReluDpVec(smallImages1)).reshape(len(inputImages),16, 7, 7)
  print("Second convolution done")
  return smallImages2

if __name__ == '__main__':
  [inputImages, inputLabels, inputIndices] = getNumbers.getImagesFromMNIST()
  smallImages = doubleConv(inputImages)
  newLabArr = np.repeat(inputLabels, 16)
  newInpArr = np.repeat(inputIndices, 16)
  showImgsOnPlt(smallImages.reshape(len(inputImages)*16, 7, 7), newLabArr, newInpArr)