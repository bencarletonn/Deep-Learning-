import os 
import cv2 
import numpy as np 
import matplotlib.pyplot as plt 

DATA_DIR = "C:/Users/benca/africananmials" # Change this to your own personal directory containing data 
CATEGORIES = ['buffalo', 'elephant', 'rhino', 'zebra'] 

dataset = []
IMG_SIZE = 100 


def createDataset(): 
    for animal in CATEGORIES:
        path = os.path.join(DATA_DIR, animal)
        class_num = CATEGORIES.index(animal) 
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                dataset.append([img_array, class_num])
            except:
                pass

createDataset()   

import random as rand
rand.shuffle(dataset)

X = [] 
y = [] 

for features, labels in dataset:
    X.append(features) 
    y.append(labels) 

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)


# Save dataset 
np.save('AfricanAnimalFeatures.npy',X) 
np.save('AfricanAnimalLabels.npy',y)