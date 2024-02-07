
import numpy as np
import os
from downPool import downPoolAndReluGPUForPassedMatrix
from getNumbers import geAllMNISTImgs, geAllMNISTTestImgs
from initWeightsAndBiases import inputArrSize, hiddenL1ArrSize, hiddenL2ArrSize, kernels2Size, kernels1Size
from convolution import convGPU

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
NUM_OF_EPOCH = 2
LEARNING_RATE = 0.01
BATCH_SIZE = 200
(IMAGES, LABELS) = geAllMNISTImgs()
MAX_INDEX = len(LABELS)
INDICES = np.arange(MAX_INDEX)
(TEST_IMAGES, TEST_LABELS) = geAllMNISTTestImgs()
LAMBDA1 = 0.01 #for input-hidden1 weights
LAMBDA2 = 0.01 #for hidden1-hidden2 weights
LAMBDA3 = 0.01 #for hidden2-output weights
#Naming convention: Lambda x divided by no. training instances

labels = None
indices = None
inputArr = None
hiddenL1Arr = None
hiddenL2Arr = None
outputArr = None
correctPosArr = None

def getWAndBDict():
		weightsAndBiasesDict = {
				"link12": None,
				"link23": None,
				"link34": None,
				"biases2": None,
				"biases3": None,
				"biases4": None,
				"kernels1": None,
				"kernels2": None,
				"kernels1Biases": None,
				"kernels2Biases": None
		}
		for key in weightsAndBiasesDict.keys():
				file = open(DIR_PATH + "/dataStore/{}.npy".format(key),'rb')
				weightsAndBiasesDict[key] = np.load(file)
				file.close()
		return weightsAndBiasesDict

weightsAndBiasesDict = getWAndBDict()

def setInitVar(startingInd):
		global  labels, correctPosArr
		labels = LABELS[startingInd: startingInd + BATCH_SIZE]
		correctPosArr = np.zeros((BATCH_SIZE,10), dtype='int')

def forwardPropagate(startingInd):
		global  matricesForPassedPixels2, matricesForPassedPixels3, inputArr, hiddenL1Arr, hiddenL2Arr, outputArr, correctPosArr
		wAndBDict = weightsAndBiasesDict
		batch = IMAGES[startingInd: startingInd + BATCH_SIZE]
		#"only passed pixels" means the pixels that "passed" relu and maxpooling
		(filteredImgs, matricesForPassedPixels, smallImages) = convGPU(batch, wAndBDict["kernels1"], wAndBDict["kernels1Biases"])
		#matricesForPassedPixels is for all image pixels that passed first convolution (and its neighbors)
		(filteredImgs2, matricesForPassedPixels2, smallImages2) = convGPU(smallImages, wAndBDict["kernels2"], wAndBDict["kernels2Biases"])
		kernelMul = kernels2Size*kernels1Size
		#matricesForPassedPixels2 is for all smallImages pixels that passed second convolution (and its neighbors)
		matricesForPassedPixels3 = downPoolAndReluGPUForPassedMatrix(np.repeat(matricesForPassedPixels, kernels2Size).reshape(BATCH_SIZE*kernelMul,14,14,3,3), filteredImgs2, np.zeros((BATCH_SIZE*kernelMul,7,7)), np.zeros((BATCH_SIZE*kernelMul, 7, 7, 3, 3), dtype=np.float32))
		#matricesForPassedPixels3 is for all image pixels that passed second convolution (and its neighbors)
		inputArr = smallImages2.reshape((BATCH_SIZE, inputArrSize))
		hiddenL1Arr = 1/(1+np.exp(-((inputArr @ wAndBDict["link12"]) + wAndBDict["biases2"])))
		hiddenL2Arr = 1/(1+np.exp(-((hiddenL1Arr @ wAndBDict["link23"]) + wAndBDict["biases3"])))
		outputArr = 1/(1+np.exp(-((hiddenL2Arr @ wAndBDict["link34"]) + wAndBDict["biases4"])))
		outputMaxPos = np.argmax(outputArr, axis=1)
		correctPosArr[np.arange(BATCH_SIZE), labels] = 1
		return np.sum(outputMaxPos == labels)/BATCH_SIZE

def backPropagate(learningRate):
	global weightsAndBiasesDict, matricesForPassedPixels2, matricesForPassedPixels3
	wAndBDict = weightsAndBiasesDict
	repeatedCalculationArr = (outputArr - correctPosArr) * outputArr * (1-outputArr)
	wAndBDict["link34"] -= (learningRate * (np.transpose(hiddenL2Arr) @ repeatedCalculationArr) + (LAMBDA3 * wAndBDict["link34"]))/BATCH_SIZE
	wAndBDict["biases4"] -= (learningRate * np.sum(repeatedCalculationArr, axis=0))//BATCH_SIZE
	repeatedCalArr2 = ((repeatedCalculationArr @ np.transpose(wAndBDict["link34"]))/10)*hiddenL2Arr*(1-hiddenL2Arr)
	wAndBDict["link23"] -= (learningRate * ((np.transpose(hiddenL1Arr) @ repeatedCalArr2) + (LAMBDA2 * wAndBDict["link23"])))/BATCH_SIZE
	wAndBDict["biases3"] -= (learningRate * np.sum(repeatedCalArr2, axis=0))/BATCH_SIZE
	repeatedCalArr3 = ((repeatedCalArr2 @ np.transpose(wAndBDict["link23"]))/hiddenL2ArrSize)*hiddenL1Arr*(1-hiddenL1Arr)
	wAndBDict["link12"] -= (learningRate * ((np.transpose(inputArr) @ repeatedCalArr3) + (LAMBDA1 * wAndBDict["link12"])))/BATCH_SIZE
	wAndBDict["biases2"] -= (learningRate * np.sum(repeatedCalArr3, axis=0))/BATCH_SIZE
	kernelMul = kernels2Size*kernels1Size
	dl_da = ((repeatedCalArr3 @ np.transpose(wAndBDict["link12"]))/hiddenL1ArrSize).reshape(BATCH_SIZE,kernelMul,7,7)
	summed_dl_da = np.sum(np.sum(np.sum(dl_da,axis=0),axis=1),axis=1)/49
	repeatedCalForBiases2 = learningRate*summed_dl_da
	matricesForPassedPixels2 = matricesForPassedPixels2.reshape(BATCH_SIZE,kernelMul,7,7,3,3)
	matricesForPassedPixels3 = matricesForPassedPixels3.reshape(BATCH_SIZE,kernelMul,7,7,3,3)
	for n in range(3):
		for m in range(3):
			matricesForPassedPixels2[:,:,:,:,n,m] *= dl_da[:,:,:,:]
	repeatedCalForKernel2 = learningRate*np.sum(np.sum(np.sum(matricesForPassedPixels2, axis=0), axis=1),axis=1)/BATCH_SIZE
	repeatedCalForKernel2 /= kernels1Size
	repeatedCalForKernel1 = learningRate*np.sum(np.sum(np.sum(matricesForPassedPixels3, axis=0), axis=1),axis=1)/BATCH_SIZE
	repeatedCalForKernel1 /= kernels2Size
	kernelsAsOneVal = learningRate*np.sum(np.sum(wAndBDict["kernels2"], axis=1), axis=1)/9
	k = 0
	for i in range(kernels1Size):
		for j in range(kernels2Size):
			wAndBDict["kernels2"][j] -= repeatedCalForKernel2[k]
			wAndBDict["kernels2Biases"][j] -=  repeatedCalForBiases2[k]
			wAndBDict["kernels1"][i] -= summed_dl_da[k] * kernelsAsOneVal[j] * repeatedCalForKernel1[k]
			wAndBDict["kernels1Biases"][i] -= summed_dl_da[k] * kernelsAsOneVal[j]
			k += 1
	weightsAndBiasesDict = wAndBDict

def convAndForwardPropagationOnTestData():
		wAndBDict = weightsAndBiasesDict
		(filteredImgs, matricesForPassedPixels, smallImages) = convGPU(TEST_IMAGES, weightsAndBiasesDict["kernels1"], weightsAndBiasesDict["kernels1Biases"])
		(filteredImgs2, matricesForPassedPixels2, smallImages2) = convGPU(smallImages, weightsAndBiasesDict["kernels2"], weightsAndBiasesDict["kernels2Biases"])
		images = smallImages2.reshape((len(TEST_IMAGES), inputArrSize))
		testHiddenL1Arr = 1/(1+np.exp(-((images @ wAndBDict["link12"]) + wAndBDict["biases2"])))
		testHiddenL2Arr = 1/(1+np.exp(-((testHiddenL1Arr @ wAndBDict["link23"]) + wAndBDict["biases3"])))
		testOutputArr = 1/(1+np.exp(-((testHiddenL2Arr @ wAndBDict["link34"]) + wAndBDict["biases4"])))
		return np.sum(TEST_LABELS == np.argmax(testOutputArr, axis=1))/len(TEST_LABELS)

def doTraining():
		epoch = 1
		while epoch <= NUM_OF_EPOCH:
				curImgIndPointer = 0
				accuracy = 0
				loopCounter = 1
				while curImgIndPointer < MAX_INDEX:
						setInitVar(curImgIndPointer)
						accuracy += forwardPropagate(curImgIndPointer)
						backPropagate(LEARNING_RATE)
						curImgIndPointer += BATCH_SIZE
						print("Epoch: {:2d} | ".format(epoch) + ("█"*round(curImgIndPointer*20/MAX_INDEX)) + ("_"*round(20-(curImgIndPointer*20/MAX_INDEX))) + " {:3d}%".format(round(curImgIndPointer*100/(MAX_INDEX)))  + " | Images used: {:5d}".format(curImgIndPointer) + " | Accuracy: {:3.2f}%".format((accuracy*100)/loopCounter), end="\r")
						loopCounter += 1
				epoch += 1
				print()
		print("Accuracy with test data: {:4.2f}%".format(convAndForwardPropagationOnTestData()*100))
		if(input("Press 1 to save weights and biases: ") == "1"):
				saveValues()

def saveValues():
		wAndBDict = weightsAndBiasesDict
		for key in wAndBDict.keys():
				file = open(DIR_PATH + "/dataStore/{}.npy".format(key),'wb')
				np.save(file, wAndBDict[key])
				file.close()
				if(wAndBDict[key].ndim == 3):
						saveVal = wAndBDict[key].reshape(len(wAndBDict[key]),len(wAndBDict[key][0])**2)
				else:
						saveVal = wAndBDict[key]
				file = open(DIR_PATH + "/dataStore/{}.txt".format(key),'w')
				np.savetxt(file,  saveVal, fmt='%.3f', delimiter="\t")
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
		doTraining()