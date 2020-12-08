import pandas as pd
import os
import numpy as np
import keras
from sklearn import preprocessing
import matplotlib.pyplot as plt
from keras.layers import Dense

def scale(X, x_min, x_max):
  nom = (X - X.min(axis=0)) * (x_max - x_min)
  denom = X.max(axis=0) - X.min(axis=0)
  denom[denom == 0] = 1
  return (x_min + nom / denom,denom,X.min(axis=0))

def getData(df):
  data = df
  return data.astype(float)

def getDataFromFile(fileName):
  df = pd.read_csv(fileName, sep=';')   #Считываем файл с помощью pandas
  return getData(df)                    #Возвращаем считанные данные из файла

def getData1(df):
  data = df
  data = data.drop(columns=['Время Мск'], axis=1)
  return data.drop(index=24,axis=0).astype(float)

def getDataFromFile1(fileName):
  df = pd.read_csv(fileName, sep=';',encoding='cp1251')   #Считываем файл с помощью pandas
  return getData1(df)                                      #Возвращаем считанные данные из файла

data_temp =  getDataFromFile(os.path.join('C:\\Users\\doveva\\PycharmProjects\\Hygrogen_neural_net\\Data temp', 'Temp data.csv'))

d = 5
m = 4
y = 2001
file = str(d) + "." + str(m) + "." + str(y) + ".csv"
data_load = getDataFromFile1(os.path.join('C:\\Users\\doveva\\PycharmProjects\\Hygrogen_neural_net\\Data load', file))
data_ld = data_load
d=d+1
while y < 2020:
  while m <=12:
    while d <=31:
      file = str(d) + "." + str(m) + "." + str(y) + ".csv"
      if os.path.exists(os.path.join('C:\\Users\\doveva\\PycharmProjects\\Hygrogen_neural_net\\Data load', file)):
        data_load = getDataFromFile1(os.path.join('C:\\Users\\doveva\\PycharmProjects\\Hygrogen_neural_net\\Data load', file))
        data_ld = data_ld.append(data_load)
        print(file)
      d=d+1
    d=1
    m=m+1
  d=1
  m=1
  y=y+1

data_ld.reset_index(drop=True, inplace=True)
data = pd.concat([data_ld, data_temp],axis=1)

xLen = 720                        #Анализируем по 30 предыдущим дням
stepsForward = 1                  #Тренируем сеть для предсказания на час вперёд
xChannels = range(10)             #Используем все входные каналы
yChannels = 0                     #Предказываем только генерацию в регионе
xNormalization = -1               #Нормируем входные каналы стандартным распределением
yNormalization = -1               #Нормируем выходные каналы стандартным распределением
valLen = 72                       #Используем 72 записей для проверки0

data = data.values
(data,max,min) = scale(data,0,1)

valLen = valLen + xLen - 1 + stepsForward
xData = data[:,xChannels]
yData = data[:,yChannels]
xTrain = np.array([xData[i:i + xLen, xChannels] for i in range(xData.shape[0] - xLen + 1 - stepsForward)])
yTrain = np.array([yData[i:i + stepsForward] for i in range(xLen, yData.shape[0] + 1 - stepsForward)])
xTrainLen = xTrain.shape[0]
bias = xLen + stepsForward + 2
xVal = xTrain[xTrainLen-valLen:]
yVal = yTrain[xTrainLen-valLen:]
xTrain = xTrain[:xTrainLen - valLen - bias]
yTrain = yTrain[:xTrainLen - valLen - bias]

print(xTrain.shape)
print(yTrain.shape)
print(xVal.shape)
print(yVal.shape)

adam = keras.optimizers.Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, amsgrad=False)
rmsprop=keras.optimizers.RMSprop(lr=0.001)

Victory=keras.Sequential()

Victory.add(keras.layers.Conv1D(50, 20, input_shape = (xTrain.shape[1], xTrain.shape[2]), activation="linear"))            #- VALID
Victory.add(keras.layers.Flatten())                                                                                        #- VALID
Victory.add(keras.layers.Dense(35, activation="linear"))                                                                   #- VALID
Victory.add(keras.layers.Dense(yTrain.shape[1], activation="linear"))                                                      #- VALID

#Victory.add(keras.layers.LSTM(yTrain.shape[1], activation='sigmoid'))                                                      - LSTM model

#Victory.add(keras.layers.TimeDistributed(Dense(2)))
#Victory.add(keras.layers.MaxPooling1D(pool_size=10, padding='valid', data_format='channels_last'))
#Victory.add(keras.layers.Conv1D(25,30, activation="linear"))
#Victory.add(keras.layers.MaxPooling1D(pool_size=5, padding='valid', data_format='channels_last'))

#Victory.add(keras.layers.Dropout(0.2))


Victory.compile(loss="mse", optimizer=adam, metrics=['mae','accuracy'])

Check = keras.callbacks.ModelCheckpoint(os.path.join('C:\\Users\\doveva\\PycharmProjects\\Hygrogen_neural_net\\Result', 'Model.hdf5'), monitor='val_loss', verbose=1, save_best_only=True, mode='min')

history = Victory.fit(xTrain, yTrain, epochs=200, batch_size=336, verbose=1, validation_data=(xVal, yVal),callbacks=[Check])

plt.plot(history.history['mean_absolute_error'],
         label='Mean absolute error on training data')
plt.plot(history.history['val_mean_absolute_error'],
         label='Mean absolute error on validation data')
plt.ylabel('Mean absolute error')
plt.legend()
plt.show()

pred = Victory.predict(np.array(xVal))
original = yVal*max[0]+min[0]
predicted = pred*max[0]+min[0]

plt.plot(original, color='black', label = 'Original data')
plt.plot(predicted, color='blue', label = 'Predicted data')
plt.legend(loc='best')
plt.title('Actual and predicted')
plt.show()

print('Prediction')
print(predicted)
print('Fact')
print(original)
