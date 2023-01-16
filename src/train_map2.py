from keras.models import load_model
from ml_module.data_extraction import extract_data
from sklearn.model_selection import train_test_split
from ml_module.models.model_1 import Model_1
import numpy as np

x_train, y_train = extract_data("track2_backup")
x_train1, y_train1 = extract_data("track2_2")
x_train = np.concatenate((x_train, x_train1), axis=0)
y_train = np.concatenate((y_train, y_train1), axis=0)
x_train,x_valid,y_train,y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=6)
model = Model_1()
model.load("./src/ml_module/saved/model1_3.h5")
model.train(x_train, y_train, x_valid, y_valid)
print(model.summary())
model.save("./src/ml_module/saved/model2_5.h5")
