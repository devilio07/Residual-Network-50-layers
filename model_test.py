import numpy as np 
import cv2
import argparse
from keras.models import load_model

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help= "Path for image.")
args = vars(ap.parse_args())

# Loadong our ResNet50 model

model = load_model('resnet50_for_gestures.h5')

img  = cv2.imread(args["image"])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (64,64))
image = np.array(img)
image = np.expand_dims(image, axis = 0)

pred = model.predict(image)
print(pred)
