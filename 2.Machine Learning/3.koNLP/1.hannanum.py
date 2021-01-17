text = "아름답지만 다소 복잡하기도 한 한국어는 전세계에서 13번째로 많이 사용되는 언어입니다."
# 한나눔 형태소 분석
from konlpy.tag import Hannanum
han = Hannanum()
print(han.analyze(text))
print(han.morphs(text))
print(han.nouns(text))
