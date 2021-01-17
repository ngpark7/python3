import tensorflow.keras.utils as utils
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import numpy as np
import matplotlib.pyplot as plt
# Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 오류 없애기
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(3)

# 1. 데이터셋 준비하기 : TensorFlow 기본 제공 데이터

# 훈련셋과 시험셋 로딩
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# 훈련셋과 검증셋 분리 (총 6만개 중)
X_val = X_train[50000:] # 5만개를 훈련셋으로 사용
Y_val = Y_train[50000:] # 5만개를 훈련셋으로 사용
X_train = X_train[:50000] # 1만개를 검증셋으로 사용
Y_train = Y_train[:50000] # 1만개를 검증셋으로 사용

X_train = X_train.reshape(50000, 784).astype('float32') / 255.0
X_val = X_val.reshape(10000, 784).astype('float32') / 255.0
X_test = X_test.reshape(10000, 784).astype('float32') / 255.0

# 라벨링 전환 : 확률 리턴
Y_train = utils.to_categorical(Y_train)
Y_val = utils.to_categorical(Y_val)
Y_test = utils.to_categorical(Y_test)

# 2. 모델 구성하기 (다층 구조로 구성하면 성능은 더욱 좋아지지만 반복 횟수가 증가하기 때문에 속도가 매우 느려짐)
model = Sequential()
model.add(Dense(units=2048, input_dim=28*28, activation='relu'))
# model.add(Dense(units=1024, activation='relu'))
# model.add(Dense(units=512, activation='relu'))
# model.add(Dense(units=256, activation='relu'))
# model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

# # 3. 모델 엮기
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. 모델 학습시키기(epochs와 batch_size를 적당히 늘리면 accuracy가 증대)
hist = model.fit(X_train, Y_train, epochs=10, batch_size=5, validation_data=(X_val, Y_val))

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

plt.show()

# 6. 모델 사용하기
loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)

print('')
print('loss : ' + str(loss_and_metrics[0]))
print('accuray : ' + str(loss_and_metrics[1]))
