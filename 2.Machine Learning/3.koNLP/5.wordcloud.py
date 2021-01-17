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

# 각자 실행 가능한 코드(Jupyter Notebook 적용)

#%%
#wordcloud graph 그리기 (한글 깨짐 현상 발생)
plt.figure()
plt.imshow(wordc, interpolation='bilinear')

#%%
#wordcloud graph 그리기 (굴림체 사용)
wordc = WordCloud(background_color='white', max_words=20, font_path='c:/Windows/Fonts/gulim.ttc', relative_scaling=0.2)
wordc.generate(text)
plt.figure()
plt.imshow(wordc, interpolation="bilinear")
plt.axis('off')

#%%
#all data로 wordcloud 그리기
word_list = komoran.nouns("%r"%data)
text = ' '.join(word_list)
wordcloud = WordCloud(background_color='white', max_words=2000, font_path='c:/Windows/Fonts/gulim.ttc', relative_scaling=0.2)
wordcloud.generate(text)
plt.figure(figsize=(15, 10))
plt.imshow(wordcloud, interpolation="bilinear")
 
#%%
#불용어 사전을 추가한 wordcloud
from wordcloud import STOPWORDS
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
불용어 = STOPWORDS | ENGLISH_STOP_WORDS | set(['대통령', '국가'])
wordcloud = WordCloud(background_color='white', max_words=2000, stopwords=불용어, font_path='c:/Windows/Fonts/gulim.ttc', relative_scaling=0.2)
wordcloud.generate(text)
plt.figure(figsize=(15, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')

# %%
#Masking
from PIL import Image
import numpy as np
img = Image.open('south_korea.png').convert('RGBA') # Mask Image 지정
mask = Image.new('RGB', img.size, (255,255,255))
mask.paste(img,img)
mask = np.array(mask)
wordcloud = WordCloud(background_color='white', max_words=2000, font_path='c:/Windows/Fonts/gulim.ttc', mask=mask, random_state=42)
wordcloud.generate(text)
wordcloud.to_file('result1.png') #result1.png file로 출력
plt.figure(figsize=(15, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')

# %%
# 팔레트 변경하기
import random
def grey_color(word, font_size, position, orientation, random_state=None, **kwargs):
    return 'hsl(0, 0%%, %d%%)' % random.randint(50, 100)
from PIL import Image
img = Image.open('south_korea.png').convert('RGBA') # Mask Image 지정
mask = Image.new('RGB', img.size, (255,255,255))
mask.paste(img,img)
mask = np.array(mask)
wordcloud = WordCloud(background_color='white', max_words=2000, font_path='c:/Windows/Fonts/gulim.ttc', mask=mask, random_state=42)
wordcloud.generate(text)
wordcloud.recolor(color_func=grey_color, random_state=3)
wordcloud.to_file('result2.png')
plt.figure(figsize=(15, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
