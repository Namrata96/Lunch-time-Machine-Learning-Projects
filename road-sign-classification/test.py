from keras.models import model_from_json
import h5py
from keras.models import model_from_json
import h5py
import matplotlib.pyplot as plt
import numpy as np
import keras
import os
from skimage import io, color, exposure, transform
import cv2 as cv2
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

IMG_SIZE = 48
def preprocess_img(img):
    # Histogram normalization in v channel
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)
    # central square crop
    min_side = min(img.shape[:-1])
    centre = img.shape[0] // 2, img.shape[1] // 2
    img = img[centre[0] - min_side // 2:centre[0] + min_side // 2,centre[1] - min_side // 2:centre[1] + min_side // 2,:]
    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))
    # roll color axis to axis 0
    img = np.rollaxis(img, -1)
    return img

import pandas as pd
test = pd.read_csv('GT-final_test.csv', sep=';')

# Load test dataset
X_test = []
y_test = []
image_paths = []

for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
    
    img_path = os.path.join('GTSRB_Test/Final_Test/Images/', file_name)
    image_paths.append(img_path)
    X_test.append(preprocess_img(io.imread(img_path)))
    y_test.append(class_id)

X_test = np.array(X_test)
y_test = np.array(y_test)

# predict and evaluate
y_pred = loaded_model.predict_classes(X_test)
wrong_pred = open("wrong_pred_filenames.txt", "w")

for idx, value in enumerate(y_pred):
    if value!=y_test[idx]:
        wrong_pred.write(str(y_test[idx]) + "\n")
        wrong_pred.write(str(image_paths[idx]) + "\n")
wrong_pred.close()
acc = float(np.sum(y_pred == y_test)) / np.size(y_pred)
print("Test accuracy = {}".format(acc))
