#KNN 테스트 : 점을 이동하여 출력


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

#두 영역을 나눌 빈 리스트 생성
tmp1 = []
tmp2 = []

#영역을 나누기 위해 두 점 사이의 거리를 구하는 함수
def dist(x,y):
    return np.sqrt((x[0]-y[0])**2 +(x[1]-y[1])**2)

#각 랜덤 점과 모든 점들의 거리를 구해 가까운 쪽의 영역으로 추가
for i in data:
    if (dist(random_points[0],i) > dist(random_points[1],i)):
        tmp2.append(i)
    else:
        tmp1.append(i)

# 점 이동
sum1 = 0
sum2 = 0
for i in tmp1:
    sum1 += i[0]
    sum2 += i[1]

new_points = []
new_points.append([sum1 / len(tmp1), sum2 / len(tmp1)])
sum1 = 0
sum2 = 0
for i in tmp2:
    sum1 += i[0]
    sum2 += i[1]
new_points.append([sum1 / len(tmp2), sum2 / len(tmp2)])


#새로운 점 출력
print(new_points)

#새로운 점 그래프
for i in tmp1:
    plt.plot(i[0],i[1],'o',color='y')
for i in tmp2:
    plt.plot(i[0],i[1],'o',color='g')
plt.plot(new_points[0][0],new_points[0][1],'x',color='r')
plt.plot(new_points[1][0],new_points[1][1],'x',color='b')
plt.show()