KNN 테스트 : 검은색 점을 기준으로 나누기

import random
import numpy as np
import matplotlib.pyplot as plt

data = []
for i in range(50):
    data.append([random.randint(40, 70), random.randint(140, 180)])
    data.append([random.randint(60, 90), random.randint(160, 200)])

# 데이터 찍어보기
print(data)

# 초기 랜덤 값 2개
random_points = [[random.randint(40, 90), random.randint(140, 200)], [random.randint(40, 90), random.randint(140, 200)]]
print(random_points)

# 데이터와 랜덤 값 그래프
for i in data:
    plt.plot(i[0], i[1], 'o', color='k')
plt.plot(random_points[0][0], random_points[0][1], 'x', color='r')
plt.plot(random_points[1][0], random_points[1][1], 'x', color='b')
plt.show()