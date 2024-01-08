
from getNumbers import getImagesFromMNIST
from showNumbers import showImgsOnPlt
import numpy as np
import numba as nb

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
], dtype=np.float32)

kernelsBiases = np.array([0.1,0.5,0.2,0.3], dtype=np.float32)

@nb.guvectorize('(float32[:,:], float32[:,:,:],float32[:,:,:])','(m,m),(i,j,j)->(i,m,m)',target='cuda')
def applyKernelsGPU(X, K, Z):
  for k in range(K.shape[0]):
    for r in range(1, X.shape[0]-1):
      for c in range(1, X.shape[1]-1):
        Z[k, r, c] = 0
        for kr in range(3):
          Z[k, r, c] += (X[r-1+kr, c-1] * K[k, kr, 0]) + (X[r-1+kr, c] * K[k, kr, 1]) + (X[r-1+kr, c+1] * K[k, kr, 2])
      Z[k, r, 0] = (X[r-1+kr, 0] * K[k, kr, 1]) + (X[r-1+kr, 1] * K[k, kr, 2])
      Z[k, r, X.shape[1]-1] = (X[r-1+kr, X.shape[1]-2] * K[k, kr, 0]) + (X[r-1+kr, X.shape[1]-1] * K[k, kr, 1])
    #for first row
    r = 0
    for c in range(1, X.shape[1]-1):
      Z[k, r, c] = 0
      for kr in range(1,3):
        Z[k, r, c] += (X[r-1+kr, c-1] * K[k, kr, 0]) + (X[r-1+kr, c] * K[k, kr, 1]) + (X[r-1+kr, c+1] * K[k, kr, 2])
    Z[k, r, 0] = (X[r-1+kr, 0] * K[k, kr, 1]) + (X[r-1+kr, 1] * K[k, kr, 2])
    Z[k, r, X.shape[1]-1] = (X[r-1+kr, X.shape[1]-2] * K[k, kr, 0]) + (X[r-1+kr, X.shape[1]-1] * K[k, kr, 1])
    #for last row
    r = X.shape[0]-1
    for c in range(1, X.shape[1]-1):
      for kr in range(2):
        Z[k, r, c] += (X[r-1+kr, c-1] * K[k, kr, 0]) + (X[r-1+kr, c] * K[k, kr, 1]) + (X[r-1+kr, c+1] * K[k, kr, 2])
    Z[k, r, 0] = (X[r-1+kr, 0] * K[k, kr, 1]) + (X[r-1+kr, 1] * K[k, kr, 2])
    Z[k, r, X.shape[1]-1] = (X[r-1+kr, X.shape[1]-2] * K[k, kr, 0]) + (X[r-1+kr, X.shape[1]-1] * K[k, kr, 1])

def applyKernelsGPUWrapper(imgs, kernels):
  nImgs = len(imgs)
  nKer = len(kernels)
  imgHeight = len(imgs[0]) #Also width
  filterImages = applyKernelsGPU(imgs, kernels, np.zeros((nImgs, nKer, imgHeight, imgHeight), dtype=np.float32))
  return filterImages.reshape(nImgs*nKer, imgHeight, imgHeight)

if __name__ == '__main__':
  imgs, labels, indices, = getImagesFromMNIST()
  #applyKernelVec = np.vectorize(applyKernels, signature="(m, n) -> (i, j, k)")
  #filterImages = applyKernelVec(imgs).reshape(4 * len(imgs), 28, 28)
  filterImages = applyKernelsGPUWrapper(imgs)
  showImgsOnPlt(filterImages, np.repeat(labels, 4), np.repeat(indices, 4))
  