#kNN : 최근접 이웃 알고리즘. k 값에 의하여 결정한 분류를 새로운 데이터의 분류로 확정 (거리를 기준)
# 실습방안
# 1. numpy 모듈로 knn 구현
# 2. random 모듈로 임의의 데이터 생성
# 3. 학습 구현
# 4. 필요 모듈 : random, numpy, matplotlib
import random # 임의 데이터 생성 라이브러리
import numpy as np # knn 구현을 위한 모듈
r = [] # red. 여성으로 표시
b = [] # blue. 남성으로 표시
for i in range(50): # 여성과 남성을 구분하기 위한 기준값 설정
    r.append([random.randint(40, 70),random.randint(140, 180),1]) # 여성 (red)
    b.append([random.randint(60, 90),random.randint(160, 200),0]) # 남성 (blue)

def distance(x,y): # 두 점 사이의 거리를 구하는 함수
    return np.sqrt(pow((x[0]-y[0]),2)+pow((x[1]-y[1]),2)) # kNN 공식 구현

def knn(x,y,k):
    result=[]
    cnt=0
    for i in range(len(y)):
        result.append([distance(x,y[i]),y[i][2]])
    result.sort() # 거리 대로 정렬
    for i in range(k):
        if (result[i][1]==1):
            cnt +=1
    if (cnt > (k/2)):
        print ("당신은 여자입니다.")
    else:
        print("당신은 남자입니다.")


# 입력창 생성
weight = input("몸무게를 입력해주세요. : ")
height = input("키를 입력해주세요. : ")
num = input("k를 입력해주세요. : ")
new = [int(weight),int(height)]
knn(new,r+b,int(num))


# 그래프로 표현하기
import matplotlib.pyplot as plt
rr = np.array(r)
bb = np.array(b)
for i,j in rr[:,:2]:
  plt.plot(i,j,'or')
for i,j in bb[:,:2]:
  plt.plot(i,j,'ob')
plt.show()
