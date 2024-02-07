
from getNumbers import getImagesFromMNIST
from showNumbers import showImgsOnPlt
import numpy as np
import numba as nb

@nb.guvectorize('(float32[:,:], float32[:,:], float32[:,:,:,:], float32[:,:,:,:], float32[:,:])','(m,m),(f,f),(n,n,i,i)->(n,n,i,i),(n,n)',target='cuda')
def downPoolAndReluGPU(img, filteredImg, dummyArr, passedMatrixImg, outputSmallImages):
  #edges are all expected to be zero, actual img is expected to be inside larger zero matrix
  for r in range(0, len(filteredImg), 2):
    for c in range(0, len(filteredImg[0]), 2):
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
      r2 = selectedR//2
      c2 = selectedC//2
      if(curVal<0):
        outputSmallImages[r2][c2] = 0.01*curVal
        for i in range(3):
          for j in range(3):
            if(i<0 or selectedR+i-1>=len(img) or j<0 or selectedC+j-1>=len(img[0])):
              continue
            passedMatrixImg[r2,c2,i,j] = 0.01*img[selectedR+i-1][selectedC+j-1]
      else:
        outputSmallImages[r2][c2] = curVal
        for i in range(3):
          for j in range(3):
            if(i<0 or selectedR+i-1>=len(img) or j<0 or selectedC+j-1>=len(img[0])):
              continue
            passedMatrixImg[r2,c2,i,j] = img[selectedR+i-1][selectedC+j-1]

@nb.guvectorize('(float32[:,:,:,:], float32[:,:], float32[:,:], float32[:,:,:,:])','(m,m,i,i),(m,m),(n,n)->(n,n,i,i)',target='cuda')
def downPoolAndReluGPUForPassedMatrix(passedMat, filteredImgs, dummyArr, outputImg):
  for r in range(0,len(filteredImgs),2):
    for c in range(0,len(filteredImgs[0]),2):
      selectedR = r
      selectedC = c
      if(filteredImgs[r][c+1] > filteredImgs[r][c]):
        selectedR = r
        selectedC = c+1
      if(filteredImgs[r+1][c] > filteredImgs[r][c+1]):
        selectedR = r+1
        selectedC = c
      if(filteredImgs[r+1][c+1] > filteredImgs[r+1][c]):
        selectedR = r+1
        selectedC = c+1
      for i in range(3):
        for j in range(3):
          outputImg[selectedR//2, selectedC//2,i,j] = passedMat[selectedR, selectedC,i,j]

def downPoolAndReluGPUWrapper(imgs, filteredImgs = None):
  if(filteredImgs is None):
    filteredImgs = imgs
  dummyArr = np.zeros((len(imgs[0])//2, len(imgs[0][0])//2, 3, 3), dtype=np.float32)
  passedMatrixImg = np.zeros((len(imgs), len(imgs[0])//2, len(imgs[0][0])//2, 3, 3), dtype=np.float32)
  outputSmallImages = np.zeros((len(imgs), len(imgs[0])//2, len(imgs[0][0])//2), dtype=np.float32)
  passedMatrixImg, outputSmallImages = downPoolAndReluGPU(imgs, filteredImgs, dummyArr, passedMatrixImg, outputSmallImages)
  return passedMatrixImg, outputSmallImages

if __name__ == '__main__':
  #run this file to test down pooling
  passedMatrixImg, smallImages = downPoolAndReluGPUWrapper(np.array([[[0.8,0,0,0],
                                                                      [0,0.5,0.3,0],
                                                                      [0,0.4,0.1,0],
                                                                      [0,0,0,0]],
                                                                      [[0,0,0,0],
                                                                      [0,0.7,0.6,0],
                                                                      [0,0.1,-0.8,0],
                                                                      [0,0,0,0.9]]]))
  showImgsOnPlt(smallImages, [1,2],[1,2])
  showImgsOnPlt(passedMatrixImg[:,:,:,1,1], [1,2],[1,2])
  
