"""

"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
import numpy as np

# create some training data
train_data = np.array([[1.0,1.0]])
train_targets = np.array([2.0])
print(train_data)
for i in range(3,10000,2):
    train_data= np.append(train_data,[[i,i]],axis=0)
    train_targets= np.append(train_targets,[i+i])

# create some test data
test_data = np.array([[2.0,2.0]])
test_targets = np.array([4.0])
for i in range(4,8000,4):
    test_data = np.append(test_data,[[i,i]],axis=0)
    test_targets = np.append(test_targets,[i+i])

layer_in = keras.layers.Input(shape=(2,))
layer1 = keras.layers.Flatten()(layer_in)
layer2 = keras.layers.Dense(20, activation=tf.nn.relu)(layer1)
layer3 = keras.layers.Dense(20, activation=tf.nn.relu)(layer2)
layer4 = keras.layers.Dense(1)(layer3)

model = Model(inputs=layer_in,
                    outputs=[layer4, layer2])

model.compile(optimizer='adam', 
              loss='mse',
              metrics=['mae'])

model.fit(train_data, train_targets, epochs=1, batch_size=1)

a= np.array([[2000,3000],[4,5]])
print(model.predict(a))