#KNN 최종

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


#전체코드
def dist(x,y):
    return np.sqrt((x[0]-y[0])**2 +(x[1]-y[1])**2)

data = []
for i in range(50):
    data.append([random.randint(40, 70),random.randint(140, 180)])
    data.append([random.randint(60, 90),random.randint(160, 200)])
random_points = [[random.randint(40, 90),random.randint(140, 200)],[random.randint(40, 90),random.randint(140, 200)]]

for k in range(10):
    tmp1 = []
    tmp2 = []
    if(k):
        for i in data:
            if (dist(random_points[0],i) > dist(random_points[1],i)):
                tmp2.append(i)
            else:
                tmp1.append(i)

        sum1=0
        sum2=0
        for i in tmp1:
            sum1 +=i[0]
            sum2 +=i[1]

        new_points = []
        new_points.append([sum1/len(tmp1),sum2/len(tmp1)])
        sum1=0
        sum2=0
        for i in tmp2:
            sum1 +=i[0]
            sum2 +=i[1]
        new_points.append([sum1/len(tmp2),sum2/len(tmp2)])

#새로운 점 그래프
for i in tmp1:
    plt.plot(i[0],i[1],'o',color='y')
for i in tmp2:
    plt.plot(i[0],i[1],'o',color='g')
plt.plot(new_points[0][0],new_points[0][1],'x',color='r')
plt.plot(new_points[1][0],new_points[1][1],'x',color='b')
plt.show()