
import numpy as np
import os
from getNumbers import geAllMNISTImgs, MAX_INDEX
from initWeightsAndBiases import inputArrSize, hiddenL1ArrSize, hiddenL2ArrSize
from convolution import doubleConv

NUM_OF_EPOCH = 3
LEARNING_RATE = 0.001
BATCH_SIZE = 200
(IMAGES, LABELS) = geAllMNISTImgs()
INDICES = np.arange(MAX_INDEX)

#global Variables
dir_path = os.path.dirname(os.path.realpath(__file__))
inputArr = None
hiddenL1Arr = None
hiddenL2Arr = None
outputArr = None
correctPosArr = None
convolutedImgs = np.zeros((MAX_INDEX, inputArrSize))
labels = None
indices = None
curImgIndPointer = 0

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

def setInitVar(startingInd, curEpoch):
    global labels, indices, inputArr, hiddenL1Arr, hiddenL2Arr, outputArr, correctPosArr, convolutedImgs
    if (curEpoch == 1):
        convolutedImgs[startingInd: startingInd + BATCH_SIZE] = doubleConv(IMAGES[startingInd: startingInd + BATCH_SIZE], False, True)
    labels = LABELS[startingInd: startingInd + BATCH_SIZE]
    indices = INDICES[startingInd: startingInd + BATCH_SIZE]
    inputArr = np.ndarray((BATCH_SIZE, inputArrSize),dtype=np.float16)
    hiddenL1Arr = np.ndarray((BATCH_SIZE, hiddenL1ArrSize),dtype=np.float16)
    hiddenL2Arr = np.ndarray((BATCH_SIZE, hiddenL2ArrSize),dtype=np.float16)
    outputArr = np.ndarray((BATCH_SIZE,10), dtype=np.float16)
    correctPosArr = np.zeros((BATCH_SIZE,10), dtype='int')

def getIncorrect(i):
    global correctPosArr
    correctPosArr[i][labels[i]] = 1

def forwardPropagate(si):
    global inputArr, hiddenL1Arr, hiddenL2Arr, outputArr, correctPosArr
    wAndBDict = weightsAndBiasesDict
    inputArr = convolutedImgs[si: si + BATCH_SIZE]/255
    hiddenL1Arr = 1/(1+np.exp(-((inputArr @ wAndBDict["link12"]) + wAndBDict["biases2"])))
    hiddenL2Arr = 1/(1+np.exp(-((hiddenL1Arr @ wAndBDict["link23"]) + wAndBDict["biases3"])))
    outputArr = 1/(1+np.exp(-((hiddenL2Arr @ wAndBDict["link34"]) + wAndBDict["biases4"])))
    getIncVec = np.vectorize(getIncorrect, otypes=[None])
    getIncVec(np.arange(len(labels)))
    return np.sum(np.argmax(outputArr, axis=1) == np.argmax(correctPosArr, axis=1))/BATCH_SIZE

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
    global curImgIndPointer
    epoch = 1
    while epoch <= NUM_OF_EPOCH:
      while curImgIndPointer < MAX_INDEX:
        setInitVar(curImgIndPointer, epoch)
        accuracy = forwardPropagate(curImgIndPointer)
        backPropagate(LEARNING_RATE)
        curImgIndPointer += BATCH_SIZE
        print("Epoch: {} | ".format(epoch) + ("█"*round((epoch+curImgIndPointer+1)*20/(NUM_OF_EPOCH+MAX_INDEX))) + ("_"*round(20-((epoch+curImgIndPointer+1)*20/(NUM_OF_EPOCH+MAX_INDEX)))) + " {:3d}%".format(round((epoch+curImgIndPointer+1)*100/(NUM_OF_EPOCH+MAX_INDEX)))  + " | Images used: {:5d}".format(curImgIndPointer) + " | Accuracy: {:3.1f}%".format(accuracy*100), end="\r")
      epoch += 1
      print()
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
    file = open(dir_path + "/dataStore/arrInput.txt",'w')
    np.savetxt(file, inputArr[-1],fmt='%.3f',delimiter="\t")
    file.close()
    file = open(dir_path + "/dataStore/arrHidden1.txt.txt",'w')
    np.savetxt(file, hiddenL1Arr[-1],fmt='%.3f',delimiter="\t")
    file.close()
    file = open(dir_path + "/dataStore/arrHidden2.txt.txt",'w')
    np.savetxt(file, hiddenL2Arr[-1],fmt='%.3f',delimiter="\t")
    file.close()
    file = open(dir_path + "/dataStore/arrOutput.txt.txt",'w')
    np.savetxt(file, outputArr[-1],fmt='%.3f',delimiter="\t")
    file.close()
    print("!!!!!!!!Values Saved!!!!!!!!!!!!!")

if(__name__ == "__main__"):
    doTraining()