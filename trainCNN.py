
import numpy as np
import os
from downPool import downPoolAndReluGPUForPassedMatrix
import getNumbers
from initWeightsAndBiases import initWeightsAndBiases, inputArrSize, hiddenL1ArrSize, hiddenL2ArrSize, kernels2Size, kernels1Size
from convolution import convGPU
from numba import jit

INIT_LEARNING_RATE = 0.005
MAX_UPDATES = 100 #max number of updates before new batch
MIN_ERR = 0.01 #during updates, if loss is lower than MIN_ERR, than a new batch is used
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
LAMBDA1 = 0.01 #for input-hidden1 weights
LAMBDA2 = 0.01 #for hidden1-hidden2 weights
LAMBDA3 = 0.01 #for hidden2-output weights

labels = None
indices = None
[images, labels, indices] = getNumbers.getImagesFromMNIST()

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
	if (input("Press 1 to initialize weights and biases (otherwise use last values): ") == "1"):
		initWeightsAndBiases()
	for key in weightsAndBiasesDict.keys():
		file = open(DIR_PATH + "/dataStore/{}.npy".format(key),'rb')
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
	errorArr += np.square(outputArr[i] - correctPosArr[i])
	if (np.argmax(outputArr[i]) == labels[i]):
		numOfCorrectAns += 1
	else:
		errorIndices.append(i)

def forwardPropagateSigmoid():
	global matricesForPassedPixels2, matricesForPassedPixels3, inputArr, hiddenL1Arr, hiddenL2Arr, outputArr, errorIndices, numOfCorrectAns, errorArr
	wAndBDict = weightsAndBiasesDict
	#"only passed pixels" means the pixels that "passed" relu and maxpooling
	(filteredImgs, matricesForPassedPixels, smallImages) = convGPU(images, wAndBDict["kernels1"], wAndBDict["kernels1Biases"])
	#matricesForPassedPixels is for all image pixels that passed first convolution (and its neighbors)
	(filteredImgs2, matricesForPassedPixels2, smallImages2) = convGPU(smallImages, wAndBDict["kernels2"], wAndBDict["kernels2Biases"])
	kernelMul = kernels2Size*kernels1Size
	#matricesForPassedPixels is for all smallImages pixels that passed second convolution (and its neighbors)
	matricesForPassedPixels3 = downPoolAndReluGPUForPassedMatrix(np.repeat(matricesForPassedPixels, kernels2Size).reshape(batchSize*kernelMul,14,14,3,3), filteredImgs2, np.zeros((batchSize*kernelMul,7,7)), np.zeros((batchSize*kernelMul, 7, 7, 3, 3), dtype=np.float32))
	#matricesForPassedPixels3 is for all image pixels that passed second convolution (and its neighbors)
	inputArr = smallImages2.reshape((batchSize, inputArrSize))
	hiddenL1Arr = 1/(1+np.exp(-((inputArr @ wAndBDict["link12"]) + wAndBDict["biases2"])))
	hiddenL2Arr = 1/(1+np.exp(-((hiddenL1Arr @ wAndBDict["link23"]) + wAndBDict["biases3"])))
	outputArr = 1/(1+np.exp(-((hiddenL2Arr @ wAndBDict["link34"]) + wAndBDict["biases4"])))
	errorIndices = []
	numOfCorrectAns = 0
	errorArr = np.zeros(10, dtype=np.float32)
	getIncVec = np.vectorize(getIncorrect, otypes=[None])
	getIncVec(np.arange(len(labels)))
	link34Norm = np.sum(wAndBDict["link34"]** 2) 
	link23Norm = np.sum(wAndBDict["link23"]** 2) 
	link12Norm = np.sum(wAndBDict["link12"]** 2) 
	return (np.sum(errorArr) + (LAMBDA3*link34Norm)+(LAMBDA2*link23Norm)+(LAMBDA1*link12Norm))/(2*batchSize)

def backPropagateSigmoid(learningRate):
	global weightsAndBiasesDict, matricesForPassedPixels2, matricesForPassedPixels3
	wAndBDict = weightsAndBiasesDict
	repeatedCalArr1 = (outputArr - correctPosArr) * (outputArr) * (1-outputArr)
	wAndBDict["link34"] -= (learningRate * ((np.transpose(hiddenL2Arr) @ repeatedCalArr1)+(LAMBDA3 * wAndBDict["link34"])))/batchSize
	wAndBDict["biases4"] -= (learningRate * np.sum(repeatedCalArr1, axis=0))/batchSize
	repeatedCalArr2 = ((repeatedCalArr1 @ np.transpose(wAndBDict["link34"]))/10)*hiddenL2Arr*(1-hiddenL2Arr)
	wAndBDict["link23"] -= (learningRate * ((np.transpose(hiddenL1Arr) @ repeatedCalArr2)+(LAMBDA2 * wAndBDict["link23"])))/batchSize
	wAndBDict["biases3"] -= (learningRate * np.sum(repeatedCalArr2, axis=0))/batchSize
	repeatedCalArr3 = ((repeatedCalArr2 @ np.transpose(wAndBDict["link23"]))/hiddenL2ArrSize)*hiddenL1Arr*(1-hiddenL1Arr)
	wAndBDict["link12"] -= (learningRate * ((np.transpose(inputArr) @ repeatedCalArr3)+(LAMBDA1 * wAndBDict["link12"])))/batchSize
	wAndBDict["biases2"] -= (learningRate * np.sum(repeatedCalArr3, axis=0))/batchSize
	kernelMul = kernels2Size*kernels1Size
	dl_da = ((repeatedCalArr3 @ np.transpose(wAndBDict["link12"]))/inputArrSize).reshape(batchSize,kernelMul,7,7)
	summed_dl_da = np.sum(np.sum(np.sum(dl_da,axis=0),axis=1),axis=1)/(batchSize*49)
	repeatedCalForBiases2 = learningRate*summed_dl_da
	matricesForPassedPixels2 = matricesForPassedPixels2.reshape(batchSize,kernelMul,7,7,3,3)
	matricesForPassedPixels3 = matricesForPassedPixels3.reshape(batchSize,kernelMul,7,7,3,3)
	for n in range(3):
		for m in range(3):
			matricesForPassedPixels2[:,:,:,:,n,m] *= dl_da[:,:,:,:]
	repeatedCalForKernel2 = learningRate*np.sum(np.sum(np.sum(matricesForPassedPixels2, axis=0), axis=1),axis=1)/batchSize
	repeatedCalForKernel2 /= kernels1Size
	repeatedCalForKernel1 = learningRate*np.sum(np.sum(np.sum(matricesForPassedPixels3, axis=0), axis=1),axis=1)/batchSize
	repeatedCalForKernel1 /= kernels2Size
	kernelsAsOneVal = learningRate*np.sum(np.sum(wAndBDict["kernels2"], axis=1), axis=1)/9
	k = 0
	for i in range(kernels1Size):
		for j in range(kernels2Size):
			wAndBDict["kernels2"][j] -= repeatedCalForKernel2[k]
			wAndBDict["kernels2Biases"][j] -=  repeatedCalForBiases2[k]
			wAndBDict["kernels1"][i] -= kernelsAsOneVal[j] * summed_dl_da[k] * repeatedCalForKernel1[k]
			wAndBDict["kernels1Biases"][i] -= kernelsAsOneVal[j] * summed_dl_da[k]
			k += 1
	weightsAndBiasesDict = wAndBDict

def forwardPropagateRelu():
	global matricesForPassedPixels2, matricesForPassedPixels3, inputArr, hiddenL1Arr, hiddenL2Arr, outputArr, errorIndices, numOfCorrectAns, errorArr, hiddenL1ArrMul, hiddenL2ArrMul, outputArrMul
	wAndBDict = weightsAndBiasesDict
	#"passed pixels" means the pixels that "passed" maxpooling
	#matricesForPassedPixels is for all image pixels that passed first convolution (and its neighbors)
	(filteredImgs, matricesForPassedPixels, smallImages) = convGPU(images, wAndBDict["kernels1"], wAndBDict["kernels1Biases"])
	#matricesForPassedPixels2 is for all smallImages pixels that passed second convolution (and its neighbors)
	(filteredImgs2, matricesForPassedPixels2, smallImages2) = convGPU(smallImages, wAndBDict["kernels2"], wAndBDict["kernels2Biases"])
	kernelMul = kernels2Size*kernels1Size
	#matricesForPassedPixels3 is for all image pixels that passed second convolution (and its neighbors)
	matricesForPassedPixels3 = downPoolAndReluGPUForPassedMatrix(np.repeat(matricesForPassedPixels, kernels2Size).reshape(batchSize*kernelMul,14,14,3,3), filteredImgs2, np.zeros((batchSize*kernelMul,7,7)), np.zeros((batchSize*kernelMul, 7, 7, 3, 3), dtype=np.float32))
	inputArr = smallImages2.reshape((batchSize, inputArrSize))
	hiddenL1Arr = ((inputArr @ wAndBDict["link12"]) + wAndBDict["biases2"])/inputArrSize
	hiddenL1Arr = np.where(hiddenL1Arr<0,0.01*hiddenL1Arr,hiddenL1Arr)
	hiddenL1ArrMul = np.where(hiddenL1Arr<0,0.01,1)
	hiddenL2Arr = ((hiddenL1Arr @ wAndBDict["link23"]) + wAndBDict["biases3"])/hiddenL1ArrSize
	hiddenL2Arr = np.where(hiddenL2Arr<0,0.01*hiddenL2Arr,hiddenL2Arr)
	hiddenL2ArrMul = np.where(hiddenL2Arr<0,0.01,1)
	outputArr = ((hiddenL2Arr @ wAndBDict["link34"]) + wAndBDict["biases4"])/hiddenL2ArrSize
	outputArr = np.where(outputArr<0,0.01*outputArr,outputArr)
	outputArrMul = np.where(outputArr<0,0.01,1)
	errorIndices = []
	numOfCorrectAns = 0
	errorArr = np.zeros(10, dtype=np.float32)
	getIncVec = np.vectorize(getIncorrect, otypes=[None])
	getIncVec(np.arange(len(labels)))
	link34Norm = np.sum(wAndBDict["link34"]** 2) 
	link23Norm = np.sum(wAndBDict["link23"]** 2) 
	link12Norm = np.sum(wAndBDict["link12"]** 2) 
	return (np.sum(errorArr) + (LAMBDA3*link34Norm)+(LAMBDA2*link23Norm)+(LAMBDA1*link12Norm))/(2*batchSize)

def backPropagateRelu(learningRate):
	global weightsAndBiasesDict, matricesForPassedPixels2, matricesForPassedPixels3, hiddenL1ArrMul, hiddenL2ArrMul, outputArrMul
	wAndBDict = weightsAndBiasesDict
	repeatedCalArr1 = (outputArr - correctPosArr) * outputArrMul#/hiddenL2ArrSize
	wAndBDict["link34"] -= (learningRate * ((np.transpose(hiddenL2Arr) @ repeatedCalArr1) + (LAMBDA3*wAndBDict["link34"])))/batchSize
	wAndBDict["biases4"] -= (learningRate * np.sum(repeatedCalArr1, axis=0))/batchSize
	repeatedCalArr2 = ((repeatedCalArr1 @ np.transpose(wAndBDict["link34"]))*hiddenL2ArrMul)/10#/(10*hiddenL1ArrSize)
	wAndBDict["link23"] -= (learningRate * ((np.transpose(hiddenL1Arr) @ repeatedCalArr2) + (LAMBDA2*wAndBDict["link23"])))/batchSize
	wAndBDict["biases3"] -= (learningRate * np.sum(repeatedCalArr2, axis=0))/batchSize
	repeatedCalArr3 = ((repeatedCalArr2 @ np.transpose(wAndBDict["link23"]))*hiddenL1ArrMul)/hiddenL2ArrSize#/(hiddenL2ArrSize*inputArrSize)
	wAndBDict["link12"] -= (learningRate * ((np.transpose(inputArr) @ repeatedCalArr3) + (LAMBDA1*wAndBDict["link12"])))/batchSize
	wAndBDict["biases2"] -= (learningRate * np.sum(repeatedCalArr3, axis=0))/batchSize
	kernelMul = kernels2Size*kernels1Size
	dl_da = ((repeatedCalArr3 @ np.transpose(wAndBDict["link12"]))/hiddenL1ArrSize).reshape(batchSize,kernelMul,7,7)
	summed_dl_da = np.sum(np.sum(np.sum(dl_da,axis=0),axis=1),axis=1)/49
	repeatedCalForBiases2 = learningRate*summed_dl_da
	matricesForPassedPixels2 = matricesForPassedPixels2.reshape(batchSize,kernelMul,7,7,3,3)
	matricesForPassedPixels3 = matricesForPassedPixels3.reshape(batchSize,kernelMul,7,7,3,3)
	for n in range(3):
		for m in range(3):
			matricesForPassedPixels2[:,:,:,:,n,m] *= dl_da[:,:,:,:]
	repeatedCalForKernel2 = learningRate*np.sum(np.sum(np.sum(matricesForPassedPixels2, axis=0), axis=1),axis=1)/batchSize
	repeatedCalForKernel2 /= kernels1Size
	repeatedCalForKernel1 = learningRate*np.sum(np.sum(np.sum(matricesForPassedPixels3, axis=0), axis=1),axis=1)/batchSize
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

def doTraining():
	global images, labels, indices, errorArr
	lastLR = INIT_LEARNING_RATE
	useRelu = True
	if (input("Press 1 to use a sigmoid (else use Relu): ") == "1"):
		useRelu = False
	while True:
		if (input("Press 1 to use a new learning rate (current value: {}): ".format(lastLR)) == "1"):
			lastLR = float(input("Type new learning rate: "))
		for updates in range(MAX_UPDATES):
			if(useRelu):
				avgErr = forwardPropagateRelu()
				backPropagateRelu(lastLR)
			else:
				avgErr = forwardPropagateSigmoid()
				backPropagateSigmoid(lastLR)
			print(("█"*round((updates+1)*10/MAX_UPDATES)) + ("_"*round(10-((updates+1)*10/MAX_UPDATES))) + " {:4d}%".format(round((updates+1)*100/MAX_UPDATES)) + " | current error: {:6.5f}".format(avgErr) + " | number of correct prediction (out of {}): {} ".format(len(indices), numOfCorrectAns),end="\r")
			if(avgErr < MIN_ERR):
				print("\nMinimum error reached, ending training...")
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
			images = images[errorIndices]
			labels = labels[errorIndices]
			indices = indices[errorIndices]
		elif(input("Press 1 to repeat training with different images (from MNIST), otherwise quit: ") == "1"):
			[images, labels, indices] = getNumbers.getImagesFromMNIST()
		else:
			print("bye")
			break
		setInitVar()

def saveValues():
	global DIR_PATH
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

	#save calculated values of last image
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
	setInitVar()
	doTraining()