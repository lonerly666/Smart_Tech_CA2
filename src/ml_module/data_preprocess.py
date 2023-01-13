import cv2
from imgaug import augmenters as iaa
from keras.utils.np_utils import to_categorical

def preprocess_img(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3,3), 0)
    img = cv2.resize(img, (200,66))
    img = img/255
    return img


def data_preprocess(x_train, y_train):
    for data in x_train:
        data[0] = preprocess_img(data[0])
        data[1] = preprocess_img(data[1])
        data[2] = preprocess_img(data[2])
