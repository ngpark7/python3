import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras import datasets, layers, models
from matplotlib import pyplot as plt

# keras 데이터 가져오기

cifar10_data = tf.keras.datasets.cifar10.load_data()
((train_data, train_label), (test_data, test_label)) = cifar10_data

print("train_data_num: {0}, \ntest_data_num: {1}, \ntrain_label_num: {2}, \ntest_label_num: {3},".format(len(train_data), len(test_data), len(train_label), len(test_label)))
## 출력물(print("train_data_num: {0}, \ntest_data_num: {1}, \ntrain_label_num: {2}, \ntest_label_num: {3},".format(len(train_data), len(test_data), len(train_label), len(test_label)))
# train_data_num: 50000,
# test_data_num: 10000,
# train_label_num: 50000,
# test_label_num: 10000,
# (32, 32, 3)

## 출력물 : 깨진 사진 나옴
print(train_data[0].shape)
plt.imshow(train_data[0])
plt.show()

# 간단한 데이터 정규화
train_data, test_data = train_data / 255., test_data / 255.

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# 3. 모델 학습과정 설정하기
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. 모델 학습시키기
hist = model.fit(train_data, train_label, epochs=30, batch_size=32, validation_data=(test_data, test_label))

# 6. 모델 평가하기
loss_and_metrics = model.evaluate(train_data, train_label, batch_size=32)
print('## evaluation loss and_metrics ##')
print(loss_and_metrics)

input = Input(shape=(32, 32, 3), dtype='float32', name='input')

output = layers.ZeroPadding2D(padding=(3, 3))(input)

output = layers.Conv2D(8, (3, 3), strides=(2, 2))(output)
output = layers.BatchNormalization()(output)
output = layers.Activation('relu')(output)

output = layers.Conv2D(16, (3, 3), strides=(3, 3))(input)
output = layers.BatchNormalization()(output)
output = layers.Activation('relu')(output)

output = layers.Conv2D(64, (3, 3), strides=(3, 3))(input)
output = layers.BatchNormalization()(output)
output = layers.Activation('relu')(output)

output = layers.GlobalAveragePooling2D()(output)
predictions = layers.Dense(10, activation='softmax')(output)

mymodel = Model(input, predictions)
mymodel.summary()

mymodel.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

mymodel.fit(train_data, train_label, batch_size=32, epochs=10, validation_data=(test_data, test_label), shuffle=True)

test_loss, test_acc = mymodel.evaluate(test_data, test_label)
print("Test Accuracy: {0}".format(test_acc))

print("CNN Model의 추론 결과 :", np.argmax(mymodel.predict(test_data[0:1])))
plt.imshow(test_data[0])
plt.show()

print("이미지의 실제 라벨 : ", test_label[0])


