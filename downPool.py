
from getNumbers import getImagesFromMNIST
from showNumbers import showImgsOnPlt
import numpy as np
import numba as nb

@nb.guvectorize('(float32[:,:], float32[:,:], float32[:,:,:,:], float32[:,:,:,:], float32[:,:])','(m,m),(f,f),(n,n,i,i)->(n,n,i,i),(n,n)',target='cuda')
def downPoolAndReluGPU(img, filteredImg, dummyArr, passedMatrixImg, outputSmallImages):
  #edges are all expected to be zero, actual img is expected to be inside larger zero matrix
  for r in range(0, len(filteredImg), 2):
    for c in range(0, len(filteredImg[0]), 2):
      curVal = 0
      selectedR = 0
      selectedC = 0
      if(filteredImg[r][c] > curVal):
        curVal = filteredImg[r][c] 
        selectedR = r
        selectedC = c
      if(filteredImg[r][c+1] > curVal):
        curVal = filteredImg[r][c+1] 
        selectedR = r
        selectedC = c+1
      if(filteredImg[r+1][c] > curVal):
        curVal = filteredImg[r+1][c] 
        selectedR = r+1
        selectedC = c
      if(filteredImg[r+1][c+1] > curVal):
        curVal = filteredImg[r+1][c+1] 
        selectedR = r+1
        selectedC = c+1
      if(curVal != 0):
        r2 = selectedR//2
        c2 = selectedC//2
        outputSmallImages[r2][c2] = curVal
        for i in range(3):
          for j in range(3):
            passedMatrixImg[r2,c2,i,j] += img[selectedR+i-1][selectedC+j-1]

@nb.guvectorize('(float32[:,:,:,:], float32[:,:], float32[:,:], float32[:,:,:,:])','(m,m,i,i),(m,m),(n,n)->(n,n,i,i)',target='cuda')
def downPoolAndReluGPUForPassedMatrix(passedMat, filteredImgs, dummyArr, outputImg):
  for r in range(0,len(passedMat[0])//2,2):
    for c in range(0,len(passedMat[0][0])//2,2):
      curVal = 0
      selectedR = 0
      selectedC = 0
      if(filteredImgs[r][c] > curVal):
        curVal = filteredImgs[r][c]
        selectedR = r
        selectedC = c
      if(filteredImgs[r][c+1] > curVal):
        curVal = filteredImgs[r][c+1]
        selectedR = r
        selectedC = c+1
      if(filteredImgs[r+1][c] > curVal):
        curVal = filteredImgs[r+1][c]
        selectedR = r+1
        selectedC = c
      if(filteredImgs[r+1][c+1] > curVal):
        curVal = filteredImgs[r+1][c+1]
        selectedR = r+1
        selectedC = c+1
      if(curVal == 0):
        for i in range(3):
          for j in range(3):
            outputImg[selectedR//2, selectedC//2,i,j] = passedMat[selectedR, selectedC,i,j]

def downPoolAndReluGPUWrapper(imgs, filteredImgs = None):
  if(filteredImgs is None):
    filteredImgs = imgs
  inputArr = np.zeros((len(imgs),len(imgs[0])+2, len(imgs[0][0])+2), dtype=np.float32)
  inputArr[:,1:-1,1:-1] = imgs[:,:,:]
  dummyArr = np.zeros((len(imgs[0])//2, len(imgs[0][0])//2, 3, 3), dtype=np.float32)
  passedMatrixImg = np.zeros((len(imgs), len(imgs[0])//2, len(imgs[0][0])//2, 3, 3), dtype=np.float32)
  outputSmallImages = np.zeros((len(imgs), len(imgs[0])//2, len(imgs[0][0])//2), dtype=np.float32)
  #passedMatrixImg will use inputArr
  #outputSmallImages will use filteredImgs
  passedMatrixImg, outputSmallImages = downPoolAndReluGPU(inputArr, filteredImgs, dummyArr, passedMatrixImg, outputSmallImages)
  return passedMatrixImg, outputSmallImages

if __name__ == '__main__':
  #run this file to test down pooling
  imgs, labels, indices = getImagesFromMNIST()
  passedMatrixImg, smallImages = downPoolAndReluGPUWrapper(imgs)
  passedMatrixImg2, smallImages2 = downPoolAndReluGPUWrapper(smallImages)
  showImgsOnPlt(smallImages2, labels, indices)
  
