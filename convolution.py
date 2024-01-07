import getNumbers
from showNumbers import showImgsOnPlt
from kernels import applyKernelsGPUWrapper, kernels
import numpy as np
from downPool import downPoolAndReluGPUWrapper

def convGPU(inputImages, kernels):
  filteredImgs = applyKernelsGPUWrapper(inputImages, kernels)
  outputPassedMatrix, smallImages = downPoolAndReluGPUWrapper(np.repeat(inputImages, len(kernels), axis=0), filteredImgs)
  return filteredImgs, outputPassedMatrix, smallImages

if __name__ == '__main__':
  [inputImages, inputLabels, inputIndices] = getNumbers.getImagesFromMNIST()
  filteredImgs, outputImages, smallImages = convGPU(inputImages, kernels)
  filteredImgs2, outputImages2, smallImages2 = convGPU(smallImages, kernels)
  newLabArr = np.repeat(inputLabels, 16)
  newInpArr = np.repeat(inputIndices, 16)
  print(np.shape(smallImages2))
  print(np.shape(outputImages2))
  showImgsOnPlt(smallImages2, newLabArr, newInpArr)
