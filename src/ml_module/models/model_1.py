from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from .model import Model

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
			Dense(3)
		])
		super().__init__(model)
		super().model.compile(loss='mse', optimizer=Adam(learning_rate=0.01))

	def train(self, x_train, y_train, x_val, y_val):
		history = super().model.fit(x_train, y_train, batch_size=50, validation_data=(x_val, y_val), epochs=50, verbose=1, shuffle=1)
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.legend(['training', 'validation'])
		plt.title('Loss')
		plt.xlabel('epoch')
		plt.show()