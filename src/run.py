from ml_module.data_extraction import extract_data
from ml_module.models.model_1 import Model_1
from sklearn.model_selection import train_test_split

x_train, y_train = extract_data("track1_1")
x_train,x_valid,y_train,y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=6)
model1 = Model_1()
model1.train(x_train, y_train, x_valid, y_valid)
print(model1.summary())
model1.save("./src/ml_module/saved/model1_5.h5")
