
from threading import Timer
import numpy as np
import os
from showNumbers import showImgsOnPlt
from getNumbers import geAllMNISTImgs, geAllMNISTTestImgs, MAX_INDEX
from initWeightsAndBiases import initWeightsAndBiases, inputArrSize, hiddenL1ArrSize, hiddenL2ArrSize
from convolution import doubleConvGPU

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
NUM_OF_EPOCH = 50
LEARNING_RATE = 0.001
BATCH_SIZE = 500
(IMAGES, LABELS) = geAllMNISTImgs()
INDICES = np.arange(MAX_INDEX)
(TEST_IMAGES, TEST_LABELS) = geAllMNISTTestImgs()
TEST_IMAGES_CONV = (doubleConvGPU(TEST_IMAGES, True))/255
LAMBDA1_DIV_INS = 0.01/BATCH_SIZE #for input-hidden1 weights
LAMBDA2_DIV_INS = 0.01/BATCH_SIZE #for hidden1-hidden2 weights
LAMBDA3_DIV_INS = 0.01/BATCH_SIZE #for hidden2-output weights
#Naming convention: Lambda x divided by no. training instances

#global Variables
inputArr = None
hiddenL1Arr = None
hiddenL2Arr = None
outputArr = None
correctPosArr = None
convolutedImgs = np.zeros((MAX_INDEX, inputArrSize))
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
        file = open(DIR_PATH + "/dataStore/{}.npy".format(key),'rb')
        weightsAndBiasesDict[key] = np.load(file)
        file.close()
    return weightsAndBiasesDict

weightsAndBiasesDict = getWAndBDict()

def setInitVar(startingInd, curEpoch):
    global labels, indices, inputArr, hiddenL1Arr, hiddenL2Arr, outputArr, correctPosArr, convolutedImgs
    if (curEpoch == 1):
        convolutedImgs[startingInd: startingInd + BATCH_SIZE] = doubleConvGPU(IMAGES[startingInd: startingInd + BATCH_SIZE], True)
    labels = LABELS[startingInd: startingInd + BATCH_SIZE]
    indices = INDICES[startingInd: startingInd + BATCH_SIZE]
    inputArr = np.ndarray((BATCH_SIZE, inputArrSize),dtype=np.float16)
    hiddenL1Arr = np.ndarray((BATCH_SIZE, hiddenL1ArrSize),dtype=np.float16)
    hiddenL2Arr = np.ndarray((BATCH_SIZE, hiddenL2ArrSize),dtype=np.float16)
    outputArr = np.ndarray((BATCH_SIZE,10), dtype=np.float16)
    correctPosArr = np.zeros((BATCH_SIZE,10), dtype='int')

def forwardPropagate(si):
    global inputArr, hiddenL1Arr, hiddenL2Arr, outputArr, correctPosArr
    wAndBDict = weightsAndBiasesDict
    inputArr = convolutedImgs[si: si + BATCH_SIZE]/255
    hiddenL1Arr = 1/(1+np.exp(-((inputArr @ wAndBDict["link12"]) + wAndBDict["biases2"])))
    hiddenL2Arr = 1/(1+np.exp(-((hiddenL1Arr @ wAndBDict["link23"]) + wAndBDict["biases3"])))
    outputArr = 1/(1+np.exp(-((hiddenL2Arr @ wAndBDict["link34"]) + wAndBDict["biases4"])))
    outputMaxPos = np.argmax(outputArr, axis=1)
    correctPosArr[np.arange(BATCH_SIZE), labels] = 1
    return np.sum(outputMaxPos == labels)/BATCH_SIZE

def backPropagate(learningRate):
    global weightsAndBiasesDict
    wAndBDict = weightsAndBiasesDict
    repeatedCalculationArr = learningRate * (outputArr - correctPosArr) * (outputArr) * (1-outputArr)
    wAndBDict["link34"] -= (np.transpose(hiddenL2Arr) @ repeatedCalculationArr)
    wAndBDict["link34"] -= learningRate * LAMBDA3_DIV_INS * wAndBDict["link34"]
    wAndBDict["biases4"] -= (np.ones(len(labels))/len(labels)) @ repeatedCalculationArr
    repeatedCalArr2 = ((repeatedCalculationArr @ np.transpose(wAndBDict["link34"]))/10)*hiddenL2Arr*(1-hiddenL2Arr)
    wAndBDict["link23"] -= (np.transpose(hiddenL1Arr) @ repeatedCalArr2)
    wAndBDict["link23"] -= learningRate * LAMBDA2_DIV_INS * wAndBDict["link23"]
    wAndBDict["biases3"] -= (np.ones(len(labels))/len(labels)) @ repeatedCalArr2
    repeatedCalArr3 = ((repeatedCalArr2 @ np.transpose(wAndBDict["link23"]))/hiddenL2ArrSize)*hiddenL1Arr*(1-hiddenL1Arr)
    wAndBDict["link12"] -= (np.transpose(inputArr) @ repeatedCalArr3) + (LAMBDA1_DIV_INS * wAndBDict["link12"])
    wAndBDict["link12"] -= learningRate * LAMBDA1_DIV_INS * wAndBDict["link12"]
    wAndBDict["biases2"] -= (np.ones(len(labels))/len(labels)) @ repeatedCalArr3
    weightsAndBiasesDict = wAndBDict

def convAndForwardPropagationOnTestData():
    wAndBDict = weightsAndBiasesDict
    testHiddenL1Arr = 1/(1+np.exp(-((TEST_IMAGES_CONV @ wAndBDict["link12"]) + wAndBDict["biases2"])))
    testHiddenL2Arr = 1/(1+np.exp(-((testHiddenL1Arr @ wAndBDict["link23"]) + wAndBDict["biases3"])))
    testOutputArr = 1/(1+np.exp(-((testHiddenL2Arr @ wAndBDict["link34"]) + wAndBDict["biases4"])))
    return np.sum(TEST_LABELS == np.argmax(testOutputArr, axis=1))/len(TEST_LABELS)

def doTraining():
    epoch = 1
    while epoch <= NUM_OF_EPOCH:
        curImgIndPointer = 0
        while curImgIndPointer < MAX_INDEX:
            setInitVar(curImgIndPointer, epoch)
            accuracy = forwardPropagate(curImgIndPointer)
            backPropagate(LEARNING_RATE)
            curImgIndPointer += BATCH_SIZE
            print("Epoch: {:2d} | ".format(epoch) + ("█"*round(curImgIndPointer*20/MAX_INDEX)) + ("_"*round(20-(curImgIndPointer*20/MAX_INDEX))) + " {:3d}%".format(round(curImgIndPointer*100/(MAX_INDEX)))  + " | Images used: {:5d}".format(curImgIndPointer) + " | Accuracy (with last batch): {:3.1f}%".format(accuracy*100), end="\r")
        epoch += 1
        print()
    print("Accuracy with test data: {}%".format(convAndForwardPropagationOnTestData()))
    if(input("Press 1 to save weights and biases: ") == "1"):
        saveValues()

def saveValues():
    wAndBDict = weightsAndBiasesDict
    for key in wAndBDict.keys():
        file = open(DIR_PATH + "/dataStore/{}.npy".format(key),'wb')
        np.save(file, wAndBDict[key])
        file.close()
        file = open(DIR_PATH + "/dataStore/{}.txt".format(key),'w')
        np.savetxt(file,  wAndBDict[key], fmt='%.3f', delimiter="\t")
        file.close()
    file = open(DIR_PATH + "/dataStore/arrInput.txt",'w')
    np.savetxt(file, inputArr[-1],fmt='%.3f',delimiter="\t")
    file.close()
    file = open(DIR_PATH + "/dataStore/arrHidden1.txt",'w')
    np.savetxt(file, hiddenL1Arr[-1],fmt='%.3f',delimiter="\t")
    file.close()
    file = open(DIR_PATH + "/dataStore/arrHidden2.txt",'w')
    np.savetxt(file, hiddenL2Arr[-1],fmt='%.3f',delimiter="\t")
    file.close()
    file = open(DIR_PATH + "/dataStore/arrOutput.txt",'w')
    np.savetxt(file, outputArr[-1],fmt='%.3f',delimiter="\t")
    file.close()
    print("!!!!!!!!Values Saved!!!!!!!!!!!!!")

if(__name__ == "__main__"):
    setInitVar(0, 1) #assume we are in first epoch
    t1 = Timer(1.0, doTraining)
    t1.start()
    showImgsOnPlt(convolutedImgs[:BATCH_SIZE].reshape(BATCH_SIZE* 16, 7, 7), LABELS[:BATCH_SIZE], INDICES[:BATCH_SIZE])