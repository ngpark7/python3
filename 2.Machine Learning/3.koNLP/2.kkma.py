text = "아름답지만 다소 복잡하기도 한 한국어는 전세계에서 13번째로 많이 사용되는 언어입니다."
# 꼬꼬마 형태소 분석
from konlpy.tag import Kkma
Kkma = Kkma()
print(Kkma.morphs(text))
print(Kkma.nouns(text))
print(Kkma.pos(text))
