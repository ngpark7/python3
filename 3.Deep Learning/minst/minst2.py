### 피마족 인디언 당뇨병 발병 데이터셋 실습-1 ###

import tensorflow.keras.utils as utils
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import numpy as np
import matplotlib.pyplot as plt
# Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 오류 없애기
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1. 데이터셋 준비하기
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")

# 훈련셋과 검증셋 분리 (총 6만개 중)
x_train = dataset[:700,0:8]
y_train = dataset[:700,8]
x_test = dataset[700:,0:8]
y_test = dataset[700:,8]

# 모델 구성하기 (다층 구조로 구성하면 성능은 더욱 좋아지지만 반복 횟수가 증가하기 때문에 속도가 매우 느려짐)
model = Sequential()
model.add(Dense(units=12, input_dim=8, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# 모델 엮기
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 학습시키기(epochs와 batch_size를 적당히 늘리면 accuracy가 증대)
hist = model.fit(x_test, y_test, epochs=1500, batch_size=64)

scores = model.evaluate(x_test, y_test)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))