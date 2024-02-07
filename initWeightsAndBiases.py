import numpy as np
import os

kernels1Size = 6
kernels2Size = 4
hiddenL2ArrSize = 128
hiddenL1ArrSize = 256
inputArrSize = kernels1Size * kernels2Size * 7 * 7
dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"

def initWeightsAndBiases():
    print("Initializing w's and b's")
    arr = []
    for i in range(inputArrSize):
        arr2 = np.random.rand(hiddenL1ArrSize) * 2 -1
        arr.append(arr2)
    file = open(dir_path +"dataStore/link12.npy",'wb')
    np.save(file,arr)
    file.close()
    file = open(dir_path +"dataStore/link12.txt",'w')
    np.savetxt(file,arr,fmt='%.3f',delimiter="\t")
    file.close()

    arr = []
    for i in range(hiddenL1ArrSize):
        arr2 = np.random.rand(hiddenL2ArrSize) * 2 -1
        arr.append(arr2)
    file = open(dir_path +"dataStore/link23.npy",'wb')
    np.save(file,arr)
    file.close()
    file = open(dir_path +"dataStore/link23.txt",'w')
    np.savetxt(file,arr,fmt='%.3f',delimiter="\t")
    file.close()

    arr = []
    for i in range(hiddenL2ArrSize):
        arr2 = np.random.rand(10) * 2 -1
        arr.append(arr2)
    file = open(dir_path +"dataStore/link34.npy",'wb')
    np.save(file,arr)
    file.close()
    file = open(dir_path +"dataStore/link34.txt",'w')
    np.savetxt(file,arr,fmt='%.3f',delimiter="\t")
    file.close()

    file = open(dir_path +"dataStore/biases2.npy",'wb')
    np.save(file,(np.random.rand(hiddenL1ArrSize)))
    file.close()
    file = open(dir_path +"dataStore/biases3.npy",'wb')
    np.save(file,(np.random.rand(hiddenL2ArrSize)))
    file.close()
    file = open(dir_path +"dataStore/biases4.npy",'wb')
    np.save(file,(np.random.rand(10)))
    file.close()

    file = open(dir_path +"dataStore/biases2.txt",'w')
    np.savetxt(file,(np.random.rand(hiddenL1ArrSize)))
    file.close()
    file = open(dir_path +"dataStore/biases3.txt",'w')
    np.savetxt(file,(np.random.rand(hiddenL2ArrSize)))
    file.close()
    file = open(dir_path +"dataStore/biases4.txt",'w')
    np.savetxt(file,(np.random.rand(10)))
    file.close()

    print("new Weights and Biases saved!!!")


def initKernels():
    
    kernels1 = np.random.normal(0,1,size=(kernels1Size,3,3)).astype(np.float32)
    kernels2 = np.random.normal(0,1,size=(kernels2Size,3,3)).astype(np.float32)
    file = open(dir_path +"dataStore/kernels1.npy",'wb')
    np.save(file, kernels1)
    file.close()
    file = open(dir_path +"dataStore/kernels2.npy",'wb')
    np.save(file, kernels2)
    file.close()

    file = open(dir_path +"dataStore/kernels1.txt",'w')
    np.savetxt(file, np.reshape(kernels1,(kernels1Size, 9)), fmt="%4.3f")
    file.close()
    file = open(dir_path +"dataStore/kernels2.txt",'w')
    np.savetxt(file, np.reshape(kernels2,(kernels2Size, 9)), fmt="%4.3f")
    file.close()

    kernels1Biases = np.random.normal(0,1,size=(kernels1Size,)).astype(np.float32)
    kernels2Biases = np.random.normal(0,1,size=(kernels2Size,)).astype(np.float32)
    file = open(dir_path +"dataStore/kernels1Biases.npy",'wb')
    np.save(file, kernels1Biases)
    file.close()
    file = open(dir_path +"dataStore/kernels2Biases.npy",'wb')
    np.save(file, kernels2Biases)
    file.close()

    file = open(dir_path +"dataStore/kernels1Biases.txt",'w')
    np.savetxt(file, kernels1Biases, fmt="%4.3f")
    file.close()
    file = open(dir_path +"dataStore/kernels2Biases.txt",'w')
    np.savetxt(file, kernels2Biases, fmt="%4.3f")
    file.close()
    print("new kernels saved!!!")

if(__name__ == "__main__"):
    initKernels()