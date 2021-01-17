# nltk import(품사태깅 할 때 이거 없으면 안나옴)
import nltk
nltk.download("book", quiet=True)
nltk.corpus.gutenberg.fileids()
emma = nltk.corpus.gutenberg.raw("austen-emma.txt")

# 품사태깅(POS Tagging)
from nltk.tag import pos_tag
from nltk.tokenize import RegexpTokenizer
ret = RegexpTokenizer("[a-zA-Z]{2,}")
tokens = ret.tokenize(emma[:1000])
tokens
tagged_list = pos_tag(tokens)
tagged_list
nouns_list = [t[0] for t in tagged_list if t[1] == 'NN']
#print(tagged_list)

# 정규표현식 : 품사 태깅에서 1번의 형식이 맞는지 확인
import re # Regular Expressin
pattern = re.compile('NN?') # ? = 0개 또는 1개의 문자,  + = 1개 이상의 문자, * = 0개 이상의 문자(wildcard) 
nouns_list = [t[0] for t in tagged_list
    if pattern.match(t[1])]

# TEXT Class
from nltk import Text
ret = RegexpTokenizer("[\\w]{2,}")
emma_text = Text(ret.tokenize(emma))
#emma_text.plot(20)
emma_text.concordance("Emma", lines=5)
emma_text.similar("general")
#emma_text.dispersion_plot(['Emma', 'Knightley', 'Frank', 'Jane', 'Harriet', 'Robert']) 

# FreqDist Class 
from nltk import FreqDist
stopwords = ['Mr', 'Mrs', 'Miss', 'Mr.' 'Mrs.' 'Dear'] #불용 단어처리
emma_tokens = pos_tag(emma_text)
names_list = [t[0] for t in emma_tokens if t[1] == "NNP" and t[0] not in stopwords]
emma_df_names = FreqDist(names_list)
emma_df_names # 이 부분 작동 안함
