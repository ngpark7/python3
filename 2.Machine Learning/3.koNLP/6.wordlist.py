# 기본 실행 코드
#%%
#법률 말뭉치(대한민국 헌법)
from konlpy.corpus import kolaw
c = kolaw.open('constitution.txt').read()
#print(c[:1000])

#wordcloud 시각화 전 빈도분석 실시
from konlpy.corpus import kolaw
data = kolaw.open('constitution.txt').read()
from konlpy.tag import Komoran
komoran = Komoran()
#print(komoran.nouns("%r"%data[0:1000]))

#명사들을 공백으로 처리
word_list = komoran.nouns("%r"%data[0:1000])
text = ' '.join(word_list)
#print(text)

#wordcloud 시각화
import matplotlib.pyplot as plt
%matplotlib inline
from wordcloud import WordCloud
wordc = WordCloud()
wordc.generate(text)

# 단어 빈도 수 계산
import nltk
import matplotlib.font_manager as fm
plt.figure(figsize=(12,6))
font_name = fm.FontProperties(fname='c:/Windows/Fonts/gulim.ttc').get_name()
plt.rc('font', family = font_name)
nltk.Text(word_list).plot(50)
