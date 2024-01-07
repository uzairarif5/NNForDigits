import os
import numpy as np
from convolution import convGPU
from getNumbers import getImagesFromMNIST
from initWeightsAndBiases import inputArrSize, hiddenL1ArrSize, hiddenL2ArrSize

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

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
				"kernels1Bias": None,
				"kernels2Bias": None
		}
		for key in weightsAndBiasesDict.keys():
				file = open(DIR_PATH + "/dataStore/{}.npy".format(key),'rb')
				weightsAndBiasesDict[key] = np.load(file)
				file.close()
		return weightsAndBiasesDict

wAndBDict = getWAndBDict()

def forwardPropagate(imgs):
		#"only passed pixels" means the pixels that "passed" relu and maxpooling
		(filteredImgs, matricesForPassedPixels, smallImages) = convGPU(imgs, wAndBDict["kernels1"])
		#matricesForPassedPixels is for all image pixels that passed first convolution (and its neighbors)
		(filteredImgs2, matricesForPassedPixels2, smallImages2) = convGPU(smallImages, wAndBDict["kernels2"])
		#matricesForPassedPixels3 is for all image pixels that passed second convolution (and its neighbors)
		images = smallImages2.reshape((len(imgs), inputArrSize))
		hiddenL1Arr = 1/(1+np.exp(-((images @ wAndBDict["link12"]) + wAndBDict["biases2"])))
		hiddenL2Arr = 1/(1+np.exp(-((hiddenL1Arr @ wAndBDict["link23"]) + wAndBDict["biases3"])))
		return 1/(1+np.exp(-((hiddenL2Arr @ wAndBDict["link34"]) + wAndBDict["biases4"])))

if(__name__ == "__main__"):
	images, labels, indices = getImagesFromMNIST(True)
	outputArr = forwardPropagate(images)
	for i in range(len(outputArr)):
		print("correctVal: {} |".format(labels[i]), end=" ")
		for num in outputArr[i]:
			print("{:4.3f}".format(num), end=" ")
		print("| prediction: {}".format(np.argmax(outputArr[i])))
	print("{} predictions correct".format(np.sum(np.argmax(outputArr, axis=1) == labels)))
	