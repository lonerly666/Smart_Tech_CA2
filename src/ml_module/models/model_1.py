from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense, PReLU
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from keras.models import load_model
from .model import Model
import numpy as np
from .commons import batch_generator
"""
NVIDIA model.
Input: Center image
Output: Steering angle

Iteration 1:
1. Only use track 1 for training and validation data
2. Was able to finish 1 lap in ~1:25 mins for track 1
3. Normal NVIDIA model.
"""

class Model_1(Model):
	def __init__(self):
		model = Sequential([
			Conv2D(24, (5, 5), strides=(2, 2), input_shape=(66, 200, 3), activation='elu'),	
			Conv2D(36, (5, 5), strides=(2, 2), activation='elu'),
			Conv2D(48, (5, 5), strides=(2, 2), activation='elu'),
			Conv2D(64, (3, 3), activation='elu'),
			Conv2D(64, (3, 3), activation='elu'),
			Flatten(),
			Dense(100, activation='elu'),
			Dense(50, activation='elu'),
			Dense(10, activation='elu'),
			Dense(1)
		])
		super().__init__(model)
		self.model.compile(loss='mse', optimizer=Adam(learning_rate=0.0001))

	def train(self, x_train, y_train, x_val, y_val):
		new_x_train = np.asarray([data[0] for data in x_train])
		new_x_val = np.asarray([data[0] for data in x_val])
		new_y_train = np.asarray([data[0] for data in y_train])
		new_y_val = np.asarray([data[0] for data in y_val])
		history = self.model.fit(batch_generator(new_x_train, new_y_train, 100, 1), steps_per_epoch=80, epochs=20, validation_data=batch_generator(new_x_val, new_y_val, 100, 0), validation_steps=80, verbose=1, shuffle=1)	
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.legend(['training', 'validation'])
		plt.title('Loss')
		plt.xlabel('epoch')
		plt.show()

	def load(self, path):
		self.model = load_model(path)