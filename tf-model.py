import numpy as np 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical # Could use tensorflow version of normalize 
from tensorflow.keras.callbacks import TensorBoard #py -m tensorboard.main --logdir=logs 
import matplotlib.pyplot as plt 
import time

X = np.load('AfricanAnimalFeatures.npy') 
y = np.load('AfricanAnimalLabels.npy')


# Normalize data 
X = X/255
# Convert labels to categorical
y = to_categorical(y, num_classes=4, dtype="int")

# Adjust parameters below to use as different training parameters 
conv_layers = [2]  #[1,2,3]
dense_layers = [1] #[0,1,2]
num_neurons = [64] #[32,64,128]
dropout = False 

# Construct architecture of model 
for conv_layer in conv_layers:
    for dense_layer in dense_layers:
        for num_neuron in num_neurons:

            model_name = f"{conv_layer}-conv-{dense_layer}-dense-{num_neuron}-neurons-{dropout}-dropout-{time.time()}"
            tensorboard = TensorBoard(log_dir=f"logs/{model_name}")
            model = Sequential()

            for i in range(conv_layer): # Convolution layers
                model.add(Conv2D(num_neuron, (3,3), input_shape = X.shape[1:], activation="relu"))
                model.add(MaxPooling2D(pool_size=(2,2)))

            model.add(Flatten()) # Dense layers take 1-D input
            for i in range(dense_layer): # Dense layers 
                model.add(Dense(num_neuron, activation="relu"))
            
            if droput == True: # Dropout layer
                model.add(Dropout(0.2))

            # Output layer
            model.add(Dense(4, activation="softmax"))


            model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
            model.fit(X, y, batch_size=18, validation_split=0.2, epochs=3, callbacks=[tensorboard]) 
