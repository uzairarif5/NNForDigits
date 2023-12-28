
import numpy as np
import os
import getNumbers
from initWeightsAndBiases import initWeightsAndBiases, inputArrSize, hiddenL1ArrSize, hiddenL2ArrSize
from showNumbers import showImgsOnPlt
from threading import Timer

INIT_LEARNING_RATE = 0.01
MAX_UPDATES = 50 #max number of updates before new batch
MIN_ERR = 0.01 #during updates, if loss is lower than MIN_ERR, than a new batch is used
dir_path = os.path.dirname(os.path.realpath(__file__))

images = None
labels = None
indices = None
if(input("Press 1 to use mnist\n") == "1"):
    [images, labels, indices] = getNumbers.getImagesFromMNIST()
else:
    images = getNumbers.getOwnImages()
    print(images[0])
    labels = getNumbers.getLabelsOfOwnImages()
    indices = list(range(len(labels)))

weightsAndBiasesDict = {
    "link12": None,
    "link23": None,
    "link34": None,
    "biases2": None,
    "biases3": None,
    "biases4": None,
}

def setWAndBDict(inputD: dict):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if (input("Press 1 to initialize weights and biases (otherwise use last values): ") == "1"):
        initWeightsAndBiases()
    for key in inputD.keys():
        file = open(dir_path + "/dataStore/{}.npy".format(key),'rb')
        inputD[key] = np.load(file)
        file.close()
    return inputD

weightsAndBiasesDict = setWAndBDict(weightsAndBiasesDict)

def setInitVar():
    global batchSize, inputArr, hiddenL1Arr, hiddenL2Arr, outputArr, correctPosArr
    batchSize = len(labels)
    inputArr = np.ndarray((batchSize, inputArrSize),dtype=np.float16)
    hiddenL1Arr = np.ndarray((batchSize, hiddenL1ArrSize),dtype=np.float16)
    hiddenL2Arr = np.ndarray((batchSize, hiddenL2ArrSize),dtype=np.float16)
    outputArr = np.ndarray((batchSize,10), dtype=np.float16)
    correctPosArr = np.zeros((batchSize,10), dtype='int')

def forwardPropagate(printOutput: bool):
    global inputArr, hiddenL1Arr, hiddenL2Arr, outputArr, correctPosArr
    wAndBDict = weightsAndBiasesDict
    errorArr = np.zeros(10, dtype=np.float32)
    numOfCorrectAns = 0
    inputArr = images.reshape((batchSize, inputArrSize))/255
    hiddenL1Arr = 1/(1+np.exp(-((inputArr @ wAndBDict["link12"]) + wAndBDict["biases2"])))
    hiddenL2Arr = 1/(1+np.exp(-((hiddenL1Arr @ wAndBDict["link23"]) + wAndBDict["biases3"])))
    outputArr = 1/(1+np.exp(-((hiddenL2Arr @ wAndBDict["link34"]) + wAndBDict["biases4"])))
    for i in range(len(labels)):
        if printOutput:
            print("output{:<2d}:".format(i), str(["%5.4f" % x for x in outputArr[i]]).replace(",","").replace("\'",""),"myAns: {}; correctAns: {}".format(np.argmax(outputArr[i]), labels[i]))
        correctPosArr[i][labels[i]] = 1
        numOfCorrectAns += 1 if (np.argmax(outputArr[i]) == labels[i]) else 0
        errorArr += np.square(outputArr[i] - correctPosArr[i])/2
    print("number of correct answers: {}/{}".format(numOfCorrectAns, len(labels)))
    avgErr = np.sum(errorArr)/len(labels)
    print("Average Error: {}".format(avgErr))
    return avgErr

def backPropagate(learningRate):
    global weightsAndBiasesDict
    print("adjusting weights")
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
    print("weights updated")

def doTraining():
    global images, labels, indices
    lastLR = INIT_LEARNING_RATE
    while True:
        printOutput = input("Press 1 to print output array on each forward propagation: ") == "1"
        if (input("Press 1 to use a new learning rate (current value: {}): ".format(lastLR)) == "1"):
            inputVal= -1
            while(inputVal < 0 or inputVal > 1):
                inputVal = float(input("Choose num between 0 and 1: "))
            lastLR = inputVal
        for updates in range(MAX_UPDATES):
            avgErr = forwardPropagate(printOutput)
            backPropagate(lastLR)
            print("{} updates completed".format(updates+1))
            if(avgErr < MIN_ERR):
                break
        if(input("Press 1 to save weights and biases: ") == "1"):
            saveValues()
        if(input("Press 1 to repeat training with same images: ") != "1"):
            if(input("Press 1 to repeat training with different images (otherwise quit): ") == "1"):
                [images, labels, indices] = getNumbers.getImagesFromMNIST()
                setInitVar()
            else:
                print("bye")
                break

def saveValues():
    global dir_path
    wAndBDict = weightsAndBiasesDict
    for key in wAndBDict.keys():
        file = open(dir_path + "/dataStore/{}.npy".format(key),'wb')
        np.save(file, wAndBDict[key])
        file.close()
        file = open(dir_path + "/dataStore/{}.txt".format(key),'w')
        np.savetxt(file,  wAndBDict[key], fmt='%.3f', delimiter="\t")
        file.close()

    #save calculated values of last image
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
    setInitVar()
    t1 = Timer(1.0, doTraining)
    t1.start()
    showImgsOnPlt(images, labels, indices)
    
    '''
    SHAPES:
    link12 (784, 256)
    link23 (256, 128)
    link34 (128, 10)
    biases2 (256,)
    biases3 (128,)
    biases4 (10,)
    inputArr: (BATCH_SIZE, 784)
    hiddenL1Arr: (BATCH_SIZE, 256)
    hiddenL2Arr: (BATCH_SIZE, 128)
    outputArr: (BATCH_SIZE, 10)
    '''