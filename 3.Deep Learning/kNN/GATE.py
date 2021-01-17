# 논리회로 And , OR , NAND
#  w1 * x1 + w2 * x2 + b > 0  흐른다.
#  w1 * x1 + w2 * x2 + b <= 0  흐르지 않는다. 

import numpy as np

#AND GATE 함수
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7

    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0  # 흐르지 않는다.
    else:
        return 1  # 흐른다.


#OR GATE 함수
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2

    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0  # 흐르지 않는다.
    else:
        return 1  # 흐른다.

#NAND GATE 함수
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7

    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0  # 흐르지 않는다.
    else:
        return 1  # 흐른다.


#GATE 값 계산하기
print("AND")
for i in [(0, 0), (1, 0), (0, 1), (1, 1)]:
    y = AND(i[0], i[1])
    print(str(i) + " -> " + str(y))
print()
print("OR")
for i in [(0, 0), (1, 0), (0, 1), (1, 1)]:
    y = OR(i[0], i[1])
    print(str(i) + " -> " + str(y))
print()
print("NAND")
for i in [(0, 0), (1, 0), (0, 1), (1, 1)]:
    y = NAND(i[0], i[1])
    print(str(i) + " -> " + str(y))

print()
print("XOR")
for i in [(0, 0), (1, 0), (0, 1), (1, 1)]:
    s1 = NAND(i[0], i[1])
    s2 = OR(i[0], i[1])
    y = AND(s1, s2)

    print(str(i) + " -> " + str(y))

#계산 값에 필요한 w1, w2, b값 정의
data = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 1)]
data_Or = [(0, 0, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
data_Nand = [(0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 0)]
data_Xor = [(0, 0, 0), (1, 0, 1), (0, 1, 1), (1, 1, 0)]

#w1, w2, b 값을 랜덤으로 생성하여 정답을 찾기 위한 작업 (학습 x)
w1 = [0, 0, 0]
w2 = [0, 0, 0]
b = [0, 0, 0]
cnt = 0
epoch = 0


#딥러닝 실시
def model(x1, x2, w1, w2, b):
    x = np.array([x1, x2])
    w = np.array([w1, w2])

    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0  # 흐르지 않는다.
    else:
        return 1  # 흐른다.


while (1):
    epoch += 1
    cnt = 0
    w1 = [np.random.normal(), np.random.normal(), np.random.normal()]
    w2 = [np.random.normal(), np.random.normal(), np.random.normal()]
    b = [np.random.normal(), np.random.normal(), np.random.normal()]

    for i in data_Xor:
        if (i[2] != model(model(i[0], i[1], w1[0], w2[0], b[0]), model(i[0], i[1], w1[1], w2[1], b[1]), w1[2], w2[2],
                          b[2])):
            break
        else:
            cnt += 1

    if cnt == 4:
        print("epoch : ", epoch)
        print("w1 : ", w1)
        print("w2 : ", w2)
        print("b : ", b)
        break

    if (epoch % 10000) == 0:
        print(epoch, " 반복 중")
