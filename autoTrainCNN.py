
import numpy as np
import os
from getNumbers import getRandomImgsFromMNIST
from initWeightsAndBiases import inputArrSize, hiddenL1ArrSize, hiddenL2ArrSize
from convolution import doubleConv

NUM_OF_TRAINING = 200
LEARNING_RATE = 0.0015
MAX_UPDATES = 1000
MIN_ERR = 0.01
NUM_OF_IMGS = 500

#global Variables
dir_path = os.path.dirname(os.path.realpath(__file__))
batchSize = None
inputArr = None
hiddenL1Arr = None
hiddenL2Arr = None
outputArr = None
correctPosArr = None
convolutedImgs = None
labels = None
indices = None

def getWAndBDict():
    weightsAndBiasesDict = {
        "link12": None,
        "link23": None,
        "link34": None,
        "biases2": None,
        "biases3": None,
        "biases4": None
    }
    for key in weightsAndBiasesDict.keys():
        file = open(dir_path + "/dataStore/{}.npy".format(key),'rb')
        weightsAndBiasesDict[key] = np.load(file)
        file.close()
    return weightsAndBiasesDict

weightsAndBiasesDict = getWAndBDict()

def setInitVar():
    global batchSize, inputArr, hiddenL1Arr, hiddenL2Arr, outputArr, correctPosArr, convolutedImgs, labels, indices
    [images, labels, indices] = getRandomImgsFromMNIST(NUM_OF_IMGS)
    convolutedImgs = doubleConv(images)
    batchSize = len(labels)
    inputArr = np.ndarray((batchSize, inputArrSize),dtype=np.float16)
    hiddenL1Arr = np.ndarray((batchSize, hiddenL1ArrSize),dtype=np.float16)
    hiddenL2Arr = np.ndarray((batchSize, hiddenL2ArrSize),dtype=np.float16)
    outputArr = np.ndarray((batchSize,10), dtype=np.float16)
    correctPosArr = np.zeros((batchSize,10), dtype='int')

def getIncorrect(i):
    global errorIndices, numOfCorrectAns, correctPosArr, errorArr
    correctPosArr[i][labels[i]] = 1
    errorArr += np.square(outputArr[i] - correctPosArr[i])/2
    if (np.argmax(outputArr[i]) == labels[i]):
        numOfCorrectAns += 1
    else:
        errorIndices.append(i)

def forwardPropagate():
    global inputArr, hiddenL1Arr, hiddenL2Arr, outputArr, errorIndices, numOfCorrectAns, errorArr
    wAndBDict = weightsAndBiasesDict
    inputArr = convolutedImgs.reshape((batchSize, inputArrSize))/255
    hiddenL1Arr = 1/(1+np.exp(-((inputArr @ wAndBDict["link12"]) + wAndBDict["biases2"])))
    hiddenL2Arr = 1/(1+np.exp(-((hiddenL1Arr @ wAndBDict["link23"]) + wAndBDict["biases3"])))
    outputArr = 1/(1+np.exp(-((hiddenL2Arr @ wAndBDict["link34"]) + wAndBDict["biases4"])))
    errorIndices = []
    numOfCorrectAns = 0
    errorArr = np.zeros(10, dtype=np.float32)
    getIncVec = np.vectorize(getIncorrect, otypes=[None])
    getIncVec(np.arange(len(labels)))

def backPropagate(learningRate):
    global weightsAndBiasesDict
    wAndBDict = weightsAndBiasesDict
    repeatedCalculationArr = learningRate * (outputArr - correctPosArr) * (outputArr) * (1-outputArr)
    wAndBDict["link34"] -= np.transpose(hiddenL2Arr) @ repeatedCalculationArr
    wAndBDict["biases4"] -= (np.ones(len(labels))/len(labels)) @ repeatedCalculationArr
    repeatedCalArr2 = ((repeatedCalculationArr @ np.transpose(wAndBDict["link34"]))/10)*hiddenL2Arr*(1-hiddenL2Arr)
    wAndBDict["link23"] -= np.transpose(hiddenL1Arr) @ repeatedCalArr2
    wAndBDict["biases3"] -= (np.ones(len(labels))/len(labels)) @ repeatedCalArr2
    repeatedCalArr3 = ((repeatedCalArr2 @ np.transpose(wAndBDict["link23"]))/hiddenL2ArrSize)*hiddenL1Arr*(1-hiddenL1Arr)
    wAndBDict["link12"] -= np.transpose(inputArr) @ repeatedCalArr3
    wAndBDict["biases2"] -= (np.ones(len(labels))/len(labels)) @ repeatedCalArr3
    weightsAndBiasesDict = wAndBDict

def doTraining():
    global convolutedImgs, labels, indices, errorArr
    counter = 0
    while counter < NUM_OF_TRAINING:
      setInitVar()
      for updates in range(MAX_UPDATES):
          forwardPropagate()
          backPropagate(LEARNING_RATE)
          avgErr = np.sum(errorArr)/len(indices)
          print(("█"*round((updates+1)*10/MAX_UPDATES)) + ("_"*round(10-((updates+1)*10/MAX_UPDATES))) + " {:4d}%".format(round((updates+1)*100/MAX_UPDATES)) + " | current error: {:6.5f}".format(avgErr) + " | number of correct prediction (out of {}): {}".format(len(indices), numOfCorrectAns),end="\r")
          if(avgErr < MIN_ERR):
              break
      counter += 1
      print("\nNumber of trainings done so far:", counter)
    if(input("Press 1 to save weights and biases: ") == "1"):
        saveValues()

def saveValues():
    wAndBDict = weightsAndBiasesDict
    for key in wAndBDict.keys():
        file = open(dir_path + "/dataStore/{}.npy".format(key),'wb')
        np.save(file, wAndBDict[key])
        file.close()
        file = open(dir_path + "/dataStore/{}.txt".format(key),'w')
        np.savetxt(file,  wAndBDict[key], fmt='%.3f', delimiter="\t")
        file.close()
    file = open(dir_path + "/dataStore/arr12.txt",'w')
    np.savetxt(file, inputArr[-1],fmt='%.3f',delimiter="\t")
    file.close()
    file = open(dir_path + "/dataStore/arr23.txt",'w')
    np.savetxt(file, hiddenL1Arr[-1],fmt='%.3f',delimiter="\t")
    file.close()
    file = open(dir_path + "/dataStore/arr34.txt",'w')
    np.savetxt(file, hiddenL2Arr[-1],fmt='%.3f',delimiter="\t")
    file.close()
    file = open(dir_path + "/dataStore/arrOut.txt",'w')
    np.savetxt(file, outputArr[-1],fmt='%.3f',delimiter="\t")
    file.close()
    print("!!!!!!!!Values Saved!!!!!!!!!!!!!")

if(__name__ == "__main__"):
    doTraining()