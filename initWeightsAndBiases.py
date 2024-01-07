import numpy as np
import os

inputArrSize = 784 #28 x 28 (width x height of images)
hiddenL1ArrSize = 256
hiddenL2ArrSize = 128

def initWeightsAndBiases():
    dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"
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

    file = open(dir_path +"dataStore/kernels1.npy",'wb')
    np.save(file, (np.random.normal(0,1,size=(4,3,3)).astype(np.float32)))
    file.close()
    file = open(dir_path +"dataStore/kernels2.npy",'wb')
    np.save(file, (np.random.normal(0,1,size=(4,3,3)).astype(np.float32)))
    file.close()

    file = open(dir_path +"dataStore/kernels1Bias.npy",'wb')
    np.save(file, (np.random.normal(0,1,size=(4,)).astype(np.float32)))
    file.close()
    file = open(dir_path +"dataStore/kernels2Bias.npy",'wb')
    np.save(file, (np.random.normal(0,1,size=(4,)).astype(np.float32)))
    file.close()

    print("values saved!!!")


if(__name__ == "__main__"):
    initWeightsAndBiases()