
## Coding a neural network to identify hand written numbers

Lets see a description of my project files and folders:
- `initWeightAndBiases.py`:
  - Contains the function of setting weights and biases which will be used by the neural network. The function will run if you run the file.
  - Contains the sizes of the input layer (784), second hidden layer (256) and third hidden layer (128).
- `/dataStore`:
  - Stores the weights and biases made from `initWeightAndBiases.py` in txt and npy files. The npy file type is used by the neural network.
- `/ownDatasetStuff`:
  - Though I am training my neural network through the MNIST dataset, I also have my own data set (which I call ownDataset) stored in this folder.
  - Custom images are made and stored using the drawing board (not in the repository as of now).
  - `ownImages.dat` contains image information, while `ownLabels.txt` contains the appropriate label (in the same index).
  - `editDataset.py` contains the functions to get, save and delete images. Running this files allows you to input one index which will delete the appropriate image, and its label.
  - My own images are stored in a bunch of 28 x 28 arrays (of floats between 0 to 1) inside one big array. `pickle.dump` is then used to store my own images in the dat file.
- `getNumbers.py`:
  - Contains functions to get images from MNIST and ownDataset.
  - Some images are banned and will not be used. Banned images are stored in the `bannedNumIndices` array.
- `showNumber.py`:
  - Contains `showImgsOnPlt`, which shows images on a matplotlib graph.
  - Running this files allows you to see images from MNIST or ownDataset.
- `trainNN.py`:
  - The main file that contains the actual neural network.
  - Understanding the back propagation can be confusing, for more detail see `/imagesForBackPro`.
  - Run this file to see the neural network in action. While training, selected images will be shown using `showImgsOnPlt` from `showNumber.py`.
  - Several prompts get asked during runtime, some of them include:
    - "Press 1 to use mnist": Either use mnist or ownDataset.
    - "Press 1 to print output array on each forward propagation": during forward propagation, this option displays the loss array for each image.
  - Program summary:
    - Select images, either by using MNIST or ownDataset. If you choose MNIST, then either neighboring images are used (with starting index and ending index chosen by the user), or images are randomly selected (the number of images is chosen by the user).
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
- `trainCNN.py`:
  - Like `trainNN.py` but the images are convoluted instead of using them directly.

If you want to inform my about any errors, or give me any suggestions, feel free to message me at my LinkedIn (linkedin.com/in/uzair0845).

### Updates

<u>update 3.1:</u>
- The maxpooling function and the filtering function are now vectorized, so convolution is now faster.
- In `trainCNN.py`, I have changed the way the progress is shown, instead of printing a new line in each update, there is now a progress bar in shown one line. That line also shows the current error value and the number of correct predictions.
- In `trainCNN.py`, added the option to view the output layer of last input array in the training loop.
- `train.py` does not show images in matplotlib anymore.
- In `showNumbers.py` and `getNumbers.py`, I added an option to view images from pre-selected indices

<u>update 2.1:</u> Added `kernel.py`, `downPool.py`, `convolution.py` and `trainCNN.py`

<u>update 1.2:</u> Made small changes to readme.md

<u>update 1.1:</u> First main update. Commits before this were to initialize my repository.