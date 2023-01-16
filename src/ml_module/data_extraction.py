from dotenv import load_dotenv
import os
from functools import cmp_to_key
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

load_dotenv()

ROOT_DIR = os.getenv('ROOT_DIR') 
DATA_DIR = ROOT_DIR + '//data//'

def extract_data(file):
	columns=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
	data = pd.read_csv(DATA_DIR + file + '//driving_log.csv', names=columns)
	pd.set_option('display.max_columns', 10)

	data.sort_values(by='center', ascending=True)

	data['label_steering'] = data['steering'].shift(-1)
	data['label_throttle'] = data['throttle'].shift(-1)
	data['label_reverse'] = data['reverse'].shift(-1)
	data.dropna(inplace=True)

	num_bins = 25
	samples_per_bin = 500
	hist, bins = np.histogram(data['label_steering'], num_bins)
	centre = (bins[:-1] + bins[1:])*0.5
	plt.bar(centre, hist, width=0.05)
	plt.plot((np.min(data['label_steering']), np.max(data['label_steering'])), (samples_per_bin, samples_per_bin))
	plt.show()

	remove_list=[]
	print('Total data: ', len(data))

	for j in range(num_bins):
		list_ = []
		for i in range(len(data['label_steering'])):
			if bins[j] <= data['label_steering'][i] <= bins[j+1]:
				list_.append(i)
		list_ = shuffle(list_)
		list_ = list_[samples_per_bin:]
		remove_list.extend(list_)

	print("Remove: ", len(remove_list))
	data.drop(data.index[remove_list], inplace=True)
	print("Remaining: ", len(data))

	hist, bins = np.histogram(data['label_steering'], num_bins)
	plt.bar(centre, hist, width=0.05)
	plt.plot((np.min(data['label_steering']), np.max(data['label_steering'])), (samples_per_bin, samples_per_bin))
	plt.show()
	x_train = data[columns[:3]].values
	y_train = data[['label_steering', 'label_throttle', 'label_reverse']].values

	return x_train, y_train
