import getNumbers
from showNumbers import showImgsOnPlt
from kernels import applyKernels, applyKernelsGPUWrapper
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

def doubleConv(inputImages, allowPrint = True, flattenOutput = False):
  if allowPrint: print("Starting convolution...") 
  filterReluDpVec = np.vectorize(filterReluDownpool, signature="(m, n) -> (i, j, k)")
  smallImages1 = filterReluDpVec(inputImages)
  if allowPrint: print("First convolution done")
  smallImages2 = filterReluDpVec(smallImages1)
  if allowPrint: print("Second convolution done")
  if(flattenOutput):
    return smallImages2.reshape(len(inputImages), 784)
  else:
    return smallImages2.reshape(len(inputImages), 16, 7, 7)

def doubleConvGPU(inputImages, flattenOutput = False):
  downSizeVectorized = np.vectorize(downSizeByHalf, signature="(m, n) -> (i, j)")
  filterImages1 = np.clip(applyKernelsGPUWrapper(inputImages),0,255)
  smallImages1 = downSizeVectorized(filterImages1)
  filterImages2 = np.clip(applyKernelsGPUWrapper(smallImages1),0,255)
  smallImages2 = downSizeVectorized(filterImages2)
  if(flattenOutput):
    return smallImages2.reshape(len(inputImages), 784)
  else:
    return smallImages2.reshape(len(inputImages), 16, 7, 7)

if __name__ == '__main__':
  [inputImages, inputLabels, inputIndices] = getNumbers.getImagesFromMNIST()
  smallImages = doubleConvGPU(inputImages)
  newLabArr = np.repeat(inputLabels, 16)
  newInpArr = np.repeat(inputIndices, 16)
  showImgsOnPlt(smallImages.reshape(len(inputImages)*16, 7, 7), newLabArr, newInpArr)
