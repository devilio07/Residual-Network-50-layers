# Residual-Network-50-layers

In my other respository I wrote a script to collect the images using OpenCV.
I used the dataset to Train this ResNet model.

# Insight
Very deep neural networks are hard to train because of exploding and vanishing gradiendts. But using Residual networks, we basically skip few connections and connect them to say 3 layers ahead. 
Reason why it works is beacuse this skipped connection is an identity function if seen mathematically which is effortless to learn for a neural network, which means we can add a couple of layers without and problem.

## What I've used:
  1. Python
  2. OpenCV
  3. Tensorflow 1.0
  4. Keras
  
## What you should know:
I've uploaded the Juyter notebook directly from my working directory. To run you'll have to download the notebook.

 1. ResNet50.ipynb contains the Residual Network (50 layers) model.
 2. utils.py has few methods used in main notebook to load dataset and other important tools.
 3. Creating_dataset.py.ipynb is my script to convert my images into .h5 formatted dataset. (I've directly uploaded the dataset     but you can create your own by making some minor changes in the code.)
 4. model_rest.py is the script that test our model.
 NOTE : saved model was of 284 MB I'll add the download link here in the future.
 
