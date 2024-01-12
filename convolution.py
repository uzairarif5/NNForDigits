import getNumbers
from showNumbers import showImgsOnPlt
from kernels import applyKernelsGPUWrapper, presetKernels, kernelsBiases
import numpy as np
from downPool import downPoolAndReluGPUWrapper

def convGPU(inputImages, kernels, kernelsBiases):
  filteredImgs = applyKernelsGPUWrapper(inputImages, kernels)
  for i in range(len(filteredImgs)):
    filteredImgs[i,:,:] += np.tile(kernelsBiases, len(inputImages))[i]
  outputPassedMatrix, smallImages = downPoolAndReluGPUWrapper(np.repeat(inputImages, len(kernels), axis=0), filteredImgs)
  return filteredImgs, outputPassedMatrix, smallImages

if __name__ == '__main__':
  [inputImages, inputLabels, inputIndices] = getNumbers.getImagesFromMNIST()
  filteredImgs, outputImages, smallImages = convGPU(inputImages, presetKernels[2:], kernelsBiases[2:])
  filteredImgs2, outputImages2, smallImages2 = convGPU(smallImages, presetKernels, kernelsBiases)
  newLabArr = np.repeat(inputLabels, 24)
  newInpArr = np.repeat(inputIndices, 24)
  print(np.shape(smallImages2))
  print(np.shape(outputImages2))
  showImgsOnPlt(smallImages2, newLabArr, newInpArr)
