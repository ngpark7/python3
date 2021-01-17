import nltk
nltk.download("book", quiet=True)
from nltk.book import * # book list 전체 출력(wildcard)
nltk.corpus.gutenberg.fileids()

emma = nltk.corpus.gutenberg.raw("austen-emma.txt")
print(emma[:200])

from nltk.tokenize import sent_tokenize
print(sent_tokenize(emma[:1000])[3])

from nltk.tokenize import word_tokenize
print(word_tokenize(emma[50:100]))

from nltk.tokenize import RegexpTokenizer
ret = RegexpTokenizer("[\\w]+")
print(ret.tokenize(emma[50:100]))