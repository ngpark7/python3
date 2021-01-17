import tensorflow as tf # 텐서플로우 모듈
# keras 모듈 불러오기(tensorflow.keras : tensorflow 내에서 keras를 import)
import tensorflow.keras.utils as utils
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import numpy as np
import random
import matplotlib.pyplot as plt
# Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 오류 없애기
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x_data = []
for i in range(100):
  x_data.append([random.randint(40, 60),random.randint(140, 170)])
  x_data.append([random.randint(60, 90),random.randint(170, 200)])
y_data = []
for i in range(100):
  y_data.append(1)#여
  y_data.append(0)#남
# 1. 데이터셋 준비하기
X_train = np.array([x_data])
X_train = X_train.reshape(200,2)

Y_train = np.array(y_data)
Y_train = Y_train.reshape(200,)

# 모델 학습하기
model = Sequential()
model.add(Dense(20, input_dim=2, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. 모델 학습시키기
hist = model.fit(X_train, Y_train, epochs=200, batch_size=10, verbose=1)


fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
# loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
# acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

x_test = np.array([[50,150],[80,180],[75,170],[60,150],[45,155]])
x_test = x_test.reshape(5,2)
y_test = np.array([1,0,0,1,1])
scores = model.evaluate(x_test, y_test)


x_data = []
for i in range(20):
  x_data.append([random.randint(40, 60),random.randint(140, 170)])
  x_data.append([random.randint(60, 90),random.randint(170, 200)])
y_data = []
for i in range(20):
  y_data.append(1)#여
  y_data.append(0)#남

# 1. 데이터셋 준비하기
x_test = np.array([x_data])
x_test = x_test.reshape(40,2)

y_test = np.array(y_data)
y_test = y_test.reshape(40,)

yhat = model.predict_classes(x_test)
for i in range(len(x_test)):
    print(x_test[i])
    print('True : ' + str(y_test[i]) + ', Predict : ' + str(yhat[i]))
    print()