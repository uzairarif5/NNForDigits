
import numpy as np
import os
import getNumbers
from initWeightsAndBiases import initWeightsAndBiases, inputArrSize, hiddenL1ArrSize, hiddenL2ArrSize
from convolution import doubleConv

INIT_LEARNING_RATE = 0.001
MAX_UPDATES = 1000 #max number of updates before new batch
MIN_ERR = 0.01 #during updates, if loss is lower than MIN_ERR, than a new batch is used
dir_path = os.path.dirname(os.path.realpath(__file__))

convolutedImgs = None
labels = None
indices = None
#get images
if(input("Press 1 to use mnist\n") == "1"):
    [images, labels, indices] = getNumbers.getImagesFromMNIST()
    convolutedImgs = doubleConv(images)
else:
    images = getNumbers.getOwnImages()
    convolutedImgs = doubleConv(images)
    labels = getNumbers.getLabelsOfOwnImages()
    indices = list(range(len(labels)))

def getWAndBDict():
    weightsAndBiasesDict = {
        "link12": None,
        "link23": None,
        "link34": None,
        "biases2": None,
        "biases3": None,
        "biases4": None
    }
    if (input("Press 1 to initialize weights and biases (otherwise use last values): ") == "1"):
        initWeightsAndBiases()
    for key in weightsAndBiasesDict.keys():
        file = open(dir_path + "/dataStore/{}.npy".format(key),'rb')
        weightsAndBiasesDict[key] = np.load(file)
        file.close()
    return weightsAndBiasesDict

weightsAndBiasesDict = getWAndBDict()

def setInitVar():
    global batchSize, inputArr, hiddenL1Arr, hiddenL2Arr, outputArr, correctPosArr
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
    lastLR = INIT_LEARNING_RATE
    while True:
        if (input("Press 1 to use a new learning rate (current value: {}): ".format(lastLR)) == "1"):
            inputVal= -1
            while(inputVal < 0 or inputVal > 1):
                inputVal = float(input("Choose num between 0 and 1: "))
            lastLR = inputVal
        for updates in range(MAX_UPDATES):
            forwardPropagate()
            backPropagate(lastLR)
            avgErr = np.sum(errorArr)/len(indices)
            print(("█"*round((updates+1)*10/MAX_UPDATES)) + ("_"*round(10-((updates+1)*10/MAX_UPDATES))) + " {:4d}%".format(round((updates+1)*100/MAX_UPDATES)) + " | current error: {:6.5f}".format(avgErr) + " | number of correct prediction (out of {}): {}".format(len(indices), numOfCorrectAns),end="\r")
            if(avgErr < MIN_ERR):
                break
        print("\nError indices:\n", indices[errorIndices])
        print("Total:",len(indices[errorIndices]))
        if(input("Press 1 to view error labels: ") == "1"):
            print(labels[errorIndices])
        if(input("Press 1 to view output layers for last input: ") == "1"):
            for i in range(len(labels)):
                print("output{:<2d}:".format(i), str(["%5.4f" % x for x in outputArr[i]]).replace(",","").replace("\'",""),"myAns: {}; correctAns: {}".format(np.argmax(outputArr[i]), labels[i]))
        if(input("Press 1 to save weights and biases: ") == "1"):
            saveValues()
        if(input("Press 1 to repeat training with the incorrect indices: ") == "1"):
            convolutedImgs = convolutedImgs[errorIndices]
            labels = labels[errorIndices]
            indices = indices[errorIndices]
        elif(input("Press 1 to repeat training with different images (from MNIST), otherwise quit: ") == "1"):
            [images, labels, indices] = getNumbers.getImagesFromMNIST()
            convolutedImgs = doubleConv(images)
        else:
            print("bye")
            break
        setInitVar()

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
    file = open(dir_path + "/dataStore/arrInput.txt.txt",'w')
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
    setInitVar()
    doTraining()
    
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