#법률 말뭉치(대한민국 헌법)
from konlpy.corpus import kolaw
c = kolaw.open('constitution.txt').read()
print(c[:1000])

#국회 의안 말뭉치
from konlpy.corpus import kobill
d = kobill.open('1809890.txt').read()
print(d[150:300])