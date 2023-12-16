import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # MATLAB-like way of plotting
# useful scikit-learn functions
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
# for brevity import specific keras objects
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
# for repeatable results:
from tensorflow.random import set_seed
from random import seed
SEED = 3
seed(SEED)
np.random.seed(SEED)
set_seed(SEED)
# generate random data
#X, y = make_classification(n_samples=200, n_features=2,
# n_classes=2, n_informative=2, n_redundant=0, flip_y=0.01,
# random_state=1)
# data from example
X = np.array([[-1,-1],[-2,-1],[-3,-2],[1,1],[2,1],[3,2]])
y = np.array([0,0,0,1,1,1])
# define the model
model = Sequential()
model.add(Dense(10, input_shape=(2,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# display a summary of the model
model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True,
show_layer_names=True)
# compile the model
model.compile(loss='binary_crossentropy', metrics=['accuracy'])
# fit the model
model.fit(X, y, epochs=50)
# evaluate the model
loss, acc = model.evaluate(X, y)
print('Loss: {:.3f}\nAccuracy: {:.3f}'.format(loss,acc))
# make a prediction
yhat = model.predict(np.array([[3,2]]))[0,0]
print('Predicted: %s (class = %d)' %
(yhat, (yhat > 0.5).astype('int32')))
