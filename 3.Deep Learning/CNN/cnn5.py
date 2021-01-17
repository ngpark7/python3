#사전작업(모듈 인스톨)
##pip install pillow
##pip install scipy


### img
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#CNN 처리
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator #폴더 내의 데이터를 불러와서 사용할 수 있도록 처리
# Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 오류 없애기
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 랜덤시드 고정시키기
np.random.seed(3)

# 1. 데이터 생성하기
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        'hard_handwriting_shape/train', # 이미지 경로 설정
        target_size=(24, 24), # 이미지 사이즈
        batch_size=3, # 3개씩 볼 것
        class_mode='categorical') # 여러개 중에 하나 사용

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        'hard_handwriting_shape/test',
        target_size=(24, 24),
        batch_size=3,
        class_mode='categorical')

# 2. 모델 구성하기
model = Sequential() ## CNN Layer
model.add(Conv2D(32, kernel_size=(3, 3), # 필터 사이즈
                 activation='relu',
                 input_shape=(24,24,3))) # 필터 수량
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) # 크기 줄이기

model.add(Flatten()) #Dence Layer
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 3. 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. 모델 학습시키기
model.fit_generator(
        train_generator,
        steps_per_epoch=15,
        epochs=50, #50번 반복
        validation_data=test_generator,
        validation_steps=5)

# 5. 모델 평가하기
print("-- Evaluate --")
scores = model.evaluate_generator(test_generator, steps=5)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

# 6. 모델 사용하기
print("-- Predict --")
output = model.predict_generator(test_generator, steps=5)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(test_generator.class_indices)
print(output)
