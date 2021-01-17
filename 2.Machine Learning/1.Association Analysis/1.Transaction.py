#package install : pip install apyori

# CSV Transaction File Add
import csv
with open('C:/Users/515-8/Desktop/code/deep/AssociationAnalysis/basket.csv', 'r', encoding='UTF8') as cf:
    transactions = []
    r = csv.reader(cf)
    for row in r:
        transactions.append(row)
print(transactions)

# Association Analysis Policy
from apyori import apriori
rules = apriori(transactions, min_support=0.1)
results = list(rules)
print(results)

# Association Analysis Policy Load
for row in results:
    support = row[1]
    ordered_stat = row[2]
    for ordered_item in ordered_stat:
        lhs = [x for x in ordered_item[0]]
        rhs = [x for x in ordered_item[1]]
        Confidence = ordered_item[3]
        lift = ordered_item[3]
        print(lhs, rhs, support, Confidence, lift, sep="\t") # 1 이하면 연관성이 없음(음의 상관관계)
