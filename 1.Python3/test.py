from typing import List

# print(1)
# a = '안녕하세요'
# print(type(a))
# print(a[0])
# print(a[0:3])
# print(a[-1])
# print('1'+'2')

# # list 자료형
# print(("l1"))
# l1 = list()
# print(1, type(1))
# print(1)

# print("l2")
# l2 = [1,2,3]
# print(type(1))
# print(1)

# print("l3")
# l3 = [1,2,3,4,5]
# print(1)
# l3.append(6)
# print(1)

# print("l4")
# l4 = [1,2,3,4,5]
# print(type(l4))
# print(l4[0:2])
# print(l4.append(7))
# print(l4)


# 조건문
# if True:
#     print(1)
#     if True:
#         print(5)
#         if True:
#             print(6)
# elif True:
#     print(2)
# else:
#     print(3)

# print("반복문")
# l5 = [1,2,3,4]
# s5 = "1234"
# t5 = (1,2,3,4)
# d5 = {'1':1,"2":2,"3":3,"4":4}
# r5 = range(1,5)
# for i in l5:
#     print(i)

# 반복문(while)
## 1번 방법
# num = 0
# while num < 5:
#     print(num)
#     num += 1

## 2번 방법(보조제어문)
# num = 0
# while True:
#     print(num)
#     num += 1
#     if num == 5:
#         break

#keyword module
# import keyword
# print(keyword.kwlist)

#numpy module
import numpy as np
l = [1,2,3]
print(l)
n = np.array(l)
print(n)