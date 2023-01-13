import random
from imgaug import augmenters as iaa
import cv2
import matplotlib.image as mpimg
import numpy as np

def zoom(image_to_zoom):
    zoom_func = iaa.Affine(scale=(1, 1.3))
    z_image = zoom_func.augment_image(image_to_zoom)
    return z_image


def pan(image_to_pan):
    pan_func = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
    pan_image = pan_func.augment_image(image_to_pan)
    return pan_image


def img_random_brightness(image_to_brighten):
    bright_func = iaa.Multiply((0.2, 1.2))
    image_to_brighten = image_to_brighten.astype('float32')
    bright_image = bright_func.augment_image(image_to_brighten).astype("uint8")
    return bright_image


def img_random_flip(image_to_flip, steering_angle):
    # 0 - flip horizontal, 1 flip vertical, -1 combo of both
    flipped_image = cv2.flip(image_to_flip, 1)
    steering_angle = -steering_angle
    return flipped_image, steering_angle

def preprocess_img_no_imread(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img

def random_augment(image_to_augment, steering_angle):
	augment_image = mpimg.imread(image_to_augment)
	if np.random.rand() < 0.5:
		augment_image = zoom(augment_image)
	if np.random.rand() < 0.5:
		augment_image = pan(augment_image)
	if np.random.rand() < 0.5:
		augment_image = img_random_brightness(augment_image)
	if np.random.rand() < 0.5:
		augment_image, steering_angle = img_random_flip(augment_image, steering_angle)
	return augment_image, steering_angle


def batch_generator(image_paths, steering_ang, batch_size, is_training):
    while True:
        batch_img = []
        batch_steering = []
        for i in range(batch_size):
            random_index = random.randint(0, len(image_paths)-1)
            if is_training:
                im, steering = random_augment(image_paths[random_index], steering_ang[random_index])
            else:
                im = mpimg.imread(image_paths[random_index])
                steering = steering_ang[random_index]

            im = preprocess_img_no_imread(im)
            batch_img.append(im)
            batch_steering.append(steering)
        yield np.asarray(batch_img), np.asarray(batch_steering)