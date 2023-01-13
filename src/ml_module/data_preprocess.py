import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_img(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3,3), 0)
    img = cv2.resize(img, (200,66))
    img = img/255
    return img

def split_data(x_train, y_train):
    x_train,y_train,x_valid,y_valid = train_test_split(x_train, y_train, test_size=0.3, random_state=6)
    return x_train, y_train, x_valid, y_valid


def data_preprocess(x_train, y_train):
    for data in x_train:
        data[0] = preprocess_img(data[0])
        data[1] = preprocess_img(data[1])
        data[2] = preprocess_img(data[2])

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_train, y_train, x_valid, y_valid = split_data(x_train, y_train)
    return x_train, y_train, x_valid, y_valid