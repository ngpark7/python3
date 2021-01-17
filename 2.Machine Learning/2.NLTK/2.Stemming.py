# Porter Stemmer
words = ["sending", "cooking", "files", "lives", "crying", "dying"]
from nltk.stem import PorterStemmer
pst = PorterStemmer()
print(pst.stem(words[0]))
print([pst.stem(w) for w in words])

result = []
for w in words :
    result.append(pst.stem(w))

# Lancaster Stemmer
words = ["sending", "cooking", "files", "lives", "crying", "dying"]
from nltk.stem import LancasterStemmer
lst = LancasterStemmer() # object 생성
print([lst.stem(w) for w in words])

# 정규표현식(Regexp Stemmer)
words = ["sending", "cooking", "files", "lives", "crying", "dying"]
from nltk.stem import RegexpStemmer
lst = RegexpStemmer('ing')
print([lst.stem(w) for w in words])

# 스페인어 추출(Snowball Stemmer)
words2 = ['enviar', 'cocina', 'moscas', 'vidas', 'ilorar', 'morir']
from nltk.stem.snowball import SnowballStemmer
sbst = SnowballStemmer('spanish')
print([sbst.stem(w) for w in words2])

# 원형복원(WordNet Lemmatizer)
word3 = ['coocking', 'believes']
from nltk.stem.wordnet import WordNetLemmatizer
wl = WordNetLemmatizer()
print([wl.lemmatize(w) for w in word3])
print([wl.lemmatize(w, pos='v') for w in word3])
