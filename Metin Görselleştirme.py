############
#Metin Görselleştirme
###########
import pandas as pd
data= pd.read_csv("C:/Users/izzet/PycharmProjects/pythonProject/NLP/train.tsv", sep="\t")
pd.set_option('display.max_columns', None)
data.head()

data.info()

#buyuk-kucuk donusumu / Sayı ve Noktalamanın kaldırılması
data["Phrase"] = data["Phrase"].apply(lambda x: " ".join(x.lower() for x in x.split()))
data["Phrase"] =data["Phrase"].str.replace("[^\w\s]","")
data["Phrase"] = data["Phrase"].str.replace("\d","")

#Stopwords
import nltk
from nltk.corpus import stopwords
sw = stopwords.words("english")
data["Phrase"] = data["Phrase"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))

#Az kullanılan kelimelerin silinmesi
sil = pd.Series(" ".join(data["Phrase"]).split()).value_counts()[-1000:]
data["Phrase"].apply(lambda x: " ".join(x for x in x.split() if x not in sil))

######
#Terim Frekansı
######

df = data.iloc[0:60000] #işlem çok uzun sürdüğü için dfi böldüm

tf1 = (df["Phrase"]).apply(lambda x:
                             pd.value_counts(x.split(" "))).sum(axis=0).reset_index()


tf1.columns =["words","tf"]

tf1.info()

a= tf1[tf1["tf"]>1000]

a.plot.bar(x= "words", y="tf");

######
#Word Cloud
######
import numpy as np
from os import path
from PIL import image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

text = data["Phrase"][0]
wordcloud = WordCloud().generate(text)

plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

text= " ".join(i for i in data.Phrase)

##############
#Metin Görselleştirme
#############