# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 19:14:39 2020

@author: cagan
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 17:39:39 2020

@author: cagan
"""

import pandas as pd
import numpy as np

import matplotlib.pylab as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import RNN
from keras.layers import TimeDistributed
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from keras import optimizers
from keras import Model
from keras.utils import plot_model
from argparse import Namespace
import copy

import tensorflow as tf


df = pd.read_csv ("Data/train_final.csv", nrows = 100000)

#Get y values
y_loc = df.columns.get_loc("HasDetections")
y = df.iloc[:,y_loc].values

#Get MachineID so it could be dropped
id_loc = df.columns.get_loc("MachineIdentifier")
ID = df.iloc[:,id_loc].values

#Drop the values
df = df.drop(columns = ["MachineIdentifier", "HasDetections"])
x = df.values

#####
#x_up = x.reshape(7,10,500000)
#x_up = x.reshape(500000,7,10,1)
#
#x_tensor = tf.convert_to_tensor(x_up)#, dtype=None, dtype_hint=None)
#input_shape = x_tensor.shape()
#x_var = x_var.numpy()
#
#x_var = tf.placeholder(tf.float32, shape=[500000,7,10,1])
#print(x_var)
#x_var = Conv2D(2, 2, activation='relu')(x_var)
#x_var = MaxPooling2D(2)(x_var)

#########################LSTM
x_vals = x.reshape(100000,70,1)
#x_tensor = tf.convert_to_tensor(x_up)#, dtype=None, dtype_hint=None)
#input_shape = x_tensor.shape()


model = Sequential()  

model.add(LSTM(units = 70, activation = 'relu',  kernel_initializer = 'uniform', return_sequences=True))
#model.add(LSTM(units = 64, activation = 'relu',  kernel_initializer = 'uniform', return_sequences=True))
#model.add(LSTM(units = 32, activation = 'relu',  kernel_initializer = 'uniform', return_sequences=True))
#model.add(LSTM(units = 16, activation = 'relu',  kernel_initializer = 'uniform', return_sequences=True))
#model.add(LSTM(units = 8, activation = 'relu',  kernel_initializer = 'uniform', return_sequences=True))
model.add(LSTM(units = 1, activation = 'sigmoid',  kernel_initializer = 'uniform', return_sequences=False))

#model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.compile(optimizer = 'adagrad', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(x_vals, y, batch_size = 10000, epochs = 50)

model.evaluate(x_vals, y) #.67-First

model.save("first-57.h5")

model = load_model("first-57.h5")

model.summary()

y_pred = model.predict(x_vals)


###################################################################


#First Model
model = Sequential()

model.add(Dense(units = 70, activation = 'relu', kernel_initializer = 'uniform'))         #, input_shape = (70,)))
#model.add(Dropout(0.5))
#model.add(Dense(units = 70, activation = 'relu', kernel_initializer = 'uniform'))
#model.add(Dropout(0.5))
model.add(Dense(units = 64, activation = 'sigmoid', kernel_initializer = 'uniform'))
#model.add(Dropout(0.5))
#model.add(Dense(units = 32, activation = 'relu', kernel_initializer = 'uniform'))
#model.add(Dropout(0.5))
model.add(Dense(units = 32, activation = 'sigmoid', kernel_initializer = 'uniform'))
#model.add(Dropout(0.5))
model.add(Dense(units = 16, activation = 'sigmoid', kernel_initializer = 'uniform'))
#model.add(Dropout(0.5))
model.add(Dense(units = 1, activation = 'sigmoid'))


model.compile(optimizer = 'adagrad', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(x, y, batch_size = 100000, epochs = 50)

model.evaluate(x, y) #.67-First

model.save("first-57.h5")

model = load_model("first-57.h5")

model.summary()

y_pred = model.predict(x)


test_prediction = y_pred
for i in range(len(y_pred)):
    if (y_pred[i] > 0.55):
        test_prediction[i] = 1
    else:
        test_prediction[i] = 0
        
        
container = copy.copy(x)
cont = pd.DataFrame(container)
feature_names = pd.array(list(cont.columns))
feature_names = list(x.columns)  
        
        
feature_importances = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_})
        
        
        
        
        
        
        
        
############################################################################### 
############################################################################### 
############################################################################### 
        
        
##TEST CODE
df = pd.read_csv ("Data/test_final.csv")

y_loc = df.columns.get_loc("HasDetections")
y = df.iloc[:,y_loc].values

#Get MachineID so it could be dropped
id_loc = df.columns.get_loc("MachineIdentifier")
ID = df.iloc[:,id_loc].values

#Drop the values
df = df.drop(columns = ["MachineIdentifier", "HasDetections"])
x = df.values

y_pred = model.predict(x)

test_prediction = y_pred
for i in range(len(y_pred)):
    if (y_pred[i] > 0.55):
        test_prediction[i] = 1
    else:
        test_prediction[i] = 0






file = pd.DataFrame(test_prediction)
file.to_csv("Data/submission1.csv")

y = pd.read_csv ("Data/test_final.csv").columns.get_loc("MachineIdentifier")

x = pd.read_csv ("Data/submission1.csv")
x = x.iloc[:, 1].values

file = pd.DataFrame(x)
file.to_csv("Data/sample_submission.csv", header = 'HasDetections')