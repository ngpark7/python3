import tensorflow as tf
print(tf.__version__)

# 수학계산용 라이브러리(numpy) 활용
import numpy as np
## 일반 리스트 출력
print('----- 일반 리스트 출력 -----')
l1 = [1,2,3]
print(l1)
print(l1[0])
n1 = np.array(l1)
print(n1)
print(n1[0])

## 다중 리스트 출력
print('----- 다중 리스트 출력 -----')
l2 = [[1,2,3],[4,5,6],[7,8,9]]
print(l2)
print(l2[0])
n2 = np.array(l2)
print(n2)
print(n2[0])

## 다중 리스트 출력2
print('----- 행렬을 이용한 리스트 출력 -----')
l3 = [[1,2,3],[4,5,6],[7,8,9]]
print(l3*2)
n3 = np.array(l3)
print(n3*2)

