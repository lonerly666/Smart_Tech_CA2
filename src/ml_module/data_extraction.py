from dotenv import load_dotenv
import os
from functools import cmp_to_key
import matplotlib.image as mpimg

load_dotenv()

# import the images and the data
# sort the data
# set label correctly
# replace image paths with image objects

ROOT_DIR = os.getenv('ROOT_DIR') 
DATA_DIR = ROOT_DIR + '//data//'

def cmp(a, b):
	if a[0] < b[0]:
		return -1
	if a[0] > b[0]:
		return 1
	return 0

# dir_name represents the data directory, e.g., track1 contains
# data from track 1, recovery contains data related to recovering
# the position of the car to the center of the road.
def import_training_data(dir_name):
	x_train = []
	with open(DATA_DIR + dir_name + '//driving_log.csv') as f:
		for line in f.readlines():
			x_train.append(line.split(','))

	# Sort based on time.
	x_train.sort(key=cmp_to_key(cmp))

	y_train = []

	# Label for time t is the data at time t + 1.
	# For each time t, the label includes the throttle, rotation and break.
	for i in range(len(x_train)):
		if i == len(x_train) - 1:
			y_train.append(x_train[i][3:6])
		else:
			y_train.append(x_train[i + 1][3:6])

	# Convert image paths to image objects.
	for data in x_train:
		x_train[0] = mpimg.imread(x_train[0])
		x_train[1] = mpimg.imread(x_train[1])
		x_train[1] = mpimg.imread(x_train[2])

	return x_train, y_train

def extract_data():
	x_train, y_train = import_training_data('track1')
	x_temp, y_temp = import_training_data('recovery')
	x_train.extend(x_temp)
	y_train.extend(y_temp)
	return x_train, y_train