class Model:
	def __init__(self, model):
		self.model = model

	def predict(self, x):
		return self.model.predict(x)

	def summary(self):
		return self.model.summary()

	def save(self, path):
		self.model.save(path)
		