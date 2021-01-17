import tensorflow as tf # 텐서플로우 모듈
# keras 모듈 불러오기(tensorflow.keras : tensorflow 내에서 keras를 import)
import tensorflow.keras.utils as utils
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import numpy as np
import matplotlib.pyplot as plt
# Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 오류 없애기
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# %matplotlib inline : 그래프 출력(주피터 노트북)

# 1. 데이터셋 준비하기 : 2의 제곱근 구현
# 훈련 데이터의 입력(x)값
X_train = np.array([1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9])

#훈련 데이터의 출력(y)값(정답)
Y_train = np.array([2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18])

#검증 데이터의 입력(x)값
X_val = np.array([1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9])

#검증 데이터의 출력(y)값(정답)
Y_val = np.array([2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18,2,4,6,8,10,12,14,16,18])

# 라벨링 전환 : Y값의 변환
Y_train = utils.to_categorical(Y_train,19)
Y_val = utils.to_categorical(Y_val,19)

# 모델 생성하기
model = Sequential()
model.add(Dense(units=38, input_dim=1, activation='elu')) #Sequential 내에 Dense를 add(층을 쌓는 과정)으로 입력값(input_dim=1) 지정하고 값은 38개로 만듬(units)
model.add(Dense(units=19,  activation='softmax')) #19개의 값 중에서 답을 하나 찾기
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy']) # 여러개의 값 중에서 틀린 갯수를 찾음(loss='categorical_crossentropy'), 틀린 것을 찾는 과정(optimizer='sgd'), 정확도 지표(metrics=['accuracy'])

# 모델 학습시키기 (verbose=2 : 학습 상태 출력, verbose=1 : 디폴트, verbose=0 : 학습 상태 출력 안함)
hist = model.fit(X_train, Y_train, epochs=200, batch_size=1, verbose=0, validation_data=(X_val, Y_val)) # 200번 반복하고 1번씩 보기

# 그래프 구현
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()
loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')
loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
plt.show() # 그래프 출력(PyCharm)

# 모델 사용하기
X_test = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
Y_test = np.array([
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

])
loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=1) # 학습 데이터를 평가

# loss 및 accuray 출력 : 그래프를 종료하면 나옴
print('')
print('loss : ' + str(loss_and_metrics[0]))
print('accuray : ' + str(loss_and_metrics[1]))