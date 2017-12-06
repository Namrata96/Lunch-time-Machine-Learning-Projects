from keras.models import model_from_json
import h5py
import matplotlib.pyplot as plt
import numpy as np
import keras
import os
from skimage import io, color, exposure, transform
import cv2 as cv2
json_file = open('model.json', 'r') # load json and create model
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5") # load weights into new model
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
X_test = [] # Load test dataset
y_test = []
import pandas as pd
test = pd.read_csv('GT-final_test.csv', sep=';')
input_file_name = raw_input("Enter image filename: ")
img_path = os.path.join('GTSRB_Test/Final_Test/Images/', input_file_name)
X_test.append(preprocess_img(io.imread(img_path)))
for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
    if file_name == input_file_name:
        y_test.append(class_id)     
X_test = np.array(X_test)
y_test = np.array(y_test)
y_pred = loaded_model.predict(X_test).argmax() # predict and evaluate
plt.title('Prediction: %s, Actual: %s' % (y_pred, y_test))
image_out = cv2.imread(img_path)
bgr_img = cv2.cvtColor(image_out, cv2.COLOR_RGB2BGR)
plt.imshow(bgr_img)
plt.show()