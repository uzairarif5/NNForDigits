
## Coding a neural network to identify hand written numbers

Lets see a small description of my project files and folders:
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

If you want to inform my about any errors, or give me any suggestions, feel free to message me at my LinkedIn (linkedin.com/in/uzair0845).

### Updates

update 1.1: First main update. Commits before this were to initialize my repository.