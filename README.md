
## Coding a neural network to identify hand written numbers

Lets see a description of my project files and folders:
- `initWeightAndBiases.py`:
  - Contains the function of setting weights and biases which will be used by the neural network. The function will run if you run the file.
  - Contains the sizes of the input layer (784), second hidden layer (256) and third hidden layer (128).
- `/dataStore`:
  - Stores the weights and biases made from `initWeightAndBiases.py` in txt and npy files. The npy file type is used by the neural network.
- `getNumbers.py`:
  - Contains functions to get images from MNIST and ownDataset.
  - Some images are banned and will not be used. Banned images are stored in the `bannedNumIndices` array.
- `showNumber.py`:
  - Contains `showImgsOnPlt`, which shows images on a matplotlib graph.
  - Running this files allows you to see images from MNIST or ownDataset.
- `trainCNN.py`:
  - The main file that contains the actual neural network.
  - Understanding the back propagation can be confusing, for more detail see `/imagesForBackPro`.
  - Run this file to see the neural network in action. While training, selected images will be shown using `showImgsOnPlt` from `showNumber.py`.
  - Several prompts get asked during runtime, some of them include:
    - "Press 1 to initialize weights and biases": Either use the last saved values or make new ones.
    - "Press 1 to use a new learning rate"
    - "Press 1 to print output array on each forward propagation": during forward propagation, this option displays the loss array for each image.
  - Program summary:
    - Select images from MNIST. Either use neighboring images (with starting index and ending index chosen by the user), or select randomly (the number of images is chosen by the user).
    - Ask user to initialize new weights and biases, otherwise used the saved values.
    - Set init variables (input array, hidden layers array, output array, correct ans array).
    - Start training.
    - The weights and biases will keep updating [MAX_UPDATES] times using the same images as long as loss is above [MIN_ERR].
    - After training is done, the user has the option to save the weights and biases, train again using the same images or train again with different images.
- `/imagesForBackPro`:
  - Contains my notes on my back propagation formulas (helps understand the back propagation algorithm in `trainNN.py`).
- `kernel.py`:
  - For CNN, this is the file that applies a kernel to an image.
  - Running this file allows you to select an image, and see the "kerneled" images.
- `downPool.py`:
  - Contains a function to down size an image by half (using max pooling).
  - Running this files allows you the select an image and downSize it twice.
- `convolution.py`:
  - Does filtering (from `kernel.py`), and down sizing (from `downPool.py`) on a range of images.
  - Running this files allows you the select a range of images and apply convolution to them.
  - Convolution currently is very slow, any recommendations to optimize will be appreciated.
- `trainCNNEntireDataset.py`:
  - Like `trainCNN.py`, but some choices are pre-selected.
  - It chooses the first [BATCH_SIZE] images from MNIST, updates the weights once, then chooses the next [BATCH_SIZE] images, updates the weights once again, and so on, until the entire dataset is used. This is known as one epoch.
  - At the end, the user will be given the option to save the new weights and biases.
  - The first [BATCH_SIZE] images will be shown using matplotlib.
- `testImage.py`

If you want to inform my about any errors, or give me any suggestions, feel free to message me at my LinkedIn (linkedin.com/in/uzair0845).

### Updates

<u>update 6.3:</u>
- Kernels now have biases as well. They are now used in forward propagation and get updated in backward propagation.

<u>update 6.2:</u>
- Added `testImage.py`, where you can pick test images from MNIST, and test the neural network. As of this update, the neural network is around 95% accurate.

<u>update 6.1:</u>
- Kernels are now initialized randomly and get saved in `/dataStore`. I didn't delete the pre-set fixed kernels I had before in `kernels.py`, I kept them for the sake of testing. The kernels also have biases as well.
- In `trainCNN.py` and `trainCNNEntireDataset.py`, kernel values also get updated in back propagation. Also, images will be convoluted on every loop instead of doing it once at the start.
- In `convolution.py`, added `convGPU` function which is like `doubleConvGPU`, but does convolution only once. Also, removed `doubleConvGPU` function.
- In `trainCNN.py` and `trainCNNEntireDataset.py`, the learning rate used to get multiplied once to the `repeatedCalArr1` variable, but now it gets multiplied on the weights and biases individually.
- In `downPool.py`, added the `downPoolAndReluGPU` function which takes an image, and does downPool and relu vectorized (using GPU), it also returns a list 3x3 matrices, these 3x3 matrices is the sum of all the pixel values that "passed" downPool and relu, along with its neighboring pixel. For example, passing 5 28x28 image will return a 14x14 image, but also 5 3x3 matrices, the middle value of the inner 3x3 matrix is the sum of all the pixel values that passed downpool and relu and ended up in the 14x14 image, the edges of the 3x3 matrix contains the passed pixel value's neighbors. These 3x3 matrices are suppose to make back propagation easier. 
- In `downPool.py`, added the function `downPoolAndReluGPUForPassedMatrix` which takes the passed matrix and the filtered image as an input, and maxpools the passed matrix in the same indices where the filtered image get maxpooled. This is used keep track which pixel in the original 28x28 image ended up in the final 7x7 image. This is suppose to make back propagation easier.

<u>update 5.2:</u>
- In `drawingBoard.py`, the drawn image is multiplied by 255 before being convoluted, and then the filtered images are divided by 255 after convolution.
- In `trainCNNEntireDataset.py`, the accuracy now represents the average accuracy of all batches passed so far in the current epoch, instead of using only the last batch.

<u>update 5.1:</u>
- Added L2 Regularization.
- Added `applyKernelsGPU` function in `kernels.py`, which uses GPU.
- Added `doubleConvGPU` in `convolution.py` which is like `doubleConv` but uses `applyKernelsGPU`.
- Removed `trainNN.py`, which was like `trainCNN.py` but without the convolution.
- Deleted `ownDataset.py`, which used to contain my own custom dataset. That also means I removed the corresponding code in `trainCNN.py` and `getNumbers.py`.
- `trainCNNEntireDataset.py` now use test data at the end to test accuracy.

<u>update 4.1:</u>
- Deleted `autoTrainCNN.py`.
- Added `trainCNNEntireDataset.py`, where the user doesn't select images, instead the first 200 images used, then weights are updated once, then the next 200 images are used, then the weights are updated again once and so on, until the entire dataset is used.
- Added `drawingBoard.py`, which shows a graphical user interface where you can draw numbers and see the output using the last save weights and biases.
- In `trainNN.py`, `trainCNN.py` and `trainCNNEntireDataset.py`, neural network node values are now saved in `arrInput.txt`, `arrHidden1.txt`, `arrHidden2.txt` and `arrOutput.txt`, instead of `arr12.txt`, `arr23.txt`, `arr34.txt` and `arrOut.txt`.
- In `convolution.py`, the print messages indicating the start and end of convolution are now optional.
- In `showNumber.py`, getting the images is now done using the `getImagesFromMNIST` function in `getNumbers.py`.
- In `getNumbers.py`, the function `getImagesFromMNIST` return images in float32 instead of uint16.

<u>update 3.2:</u>
- Prompt in `getNumbers.py` changed from "Choose the number to use" to "Number of images to use".
- Added `autoTrainCNN.py`, which is like `trainCNN.py`, but some options are pre-selected. It chooses 1500 random images from MNIST, runs 1000 updates, then chooses 1500 random images again. This keeps repeating [NUM_OF_TRAINING] times. At the end, the user will be given the option to save the new weights ans biases.

<u>update 3.1:</u>
- The maxpooling function and the filtering function are now vectorized, so convolution is now faster.
- In `trainCNN.py`, I have changed the way the progress is shown, instead of printing a new line in each update, there is now a progress bar in shown one line. That line also shows the current error value and the number of correct predictions.
- In `trainCNN.py`, added the option to view the output layer of last input array in the training loop.
- `train.py` does not show images in matplotlib anymore.
- In `showNumbers.py` and `getNumbers.py`, I added an option to view images from pre-selected indices

<u>update 2.1:</u> Added `kernel.py`, `downPool.py`, `convolution.py` and `trainCNN.py`

<u>update 1.2:</u> Made small changes to readme.md

<u>update 1.1:</u> First main update. Commits before this were to initialize my repository.