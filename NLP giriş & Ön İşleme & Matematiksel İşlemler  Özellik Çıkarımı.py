############################
#Metin Madenciliği ve Doğal Dil İşleme Giriş
############################
#Metin Ön İşleme
############################
import nltk

metin = """"
A Scandal in Bohemia! 01
The Red-Headed League,2
A Case, of Identity 33
The Boscombe Valley Mystery4
The Five Orange Pips1
The Man with? the Twisted Lip
The Adventure of the Blue Carbuncle
The Adventure of the Speckled Band
The Adventure of the Engineer's Thumb
The Adventure of the Noble Bachelor
The Adventure of the Beryl Coronet
The Adventure of the Copper Beeches"""

##################
##Bölme /Stringi DT array ve seriye çevirmek
#####################
metin.split("\n")
v_metin=metin.split("\n")

import pandas as pd

v= pd.Series(v_metin)

metin_vektoru = v[1:len(v)]
mdf = pd.DataFrame(metin_vektoru, columns=["hikayeler"])
###Büyük Küçük Harf Dönüşüm İşlemleri

d_mdf = mdf.copy()

list1 = [1,2,3]
str1 = "".join(str(e) for e in list1)

d_mdf['hikayeler'].apply(lambda x: " ".join(x.lower() for x in x.split()))
d_mdf['hikayeler'].apply(lambda x: " ".join(x.lower() for x in x.split()))

#### Noktalama İşaretlerinin Silinmesi
d_mdf['hikayeler'] =d_mdf['hikayeler'].str.replace("[^\w\s]","")

######Sayıların Silinmesi
d_mdf['hikayeler'] = d_mdf['hikayeler'].str.replace("\d","")
d_mdf = pd.DataFrame(d_mdf, columns=["hikayeler"])
d_mdf['hikayeler'] = d_mdf['hikayeler'].apply(lambda x : " ".join(x.lower() for x in x.split()))
#Stopwords

# !pip install nltk
import nltk
from nltk.corpus import stopwords
sw = stopwords.words("english")
d_mdf["hikayeler"] = d_mdf["hikayeler"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))

#################
#Az  geçen kelimelerin siiinmesi
#################

pd.Series(" ".join(d_mdf["hikayeler"]).split()).value_counts()
sil = pd.Series(" ".join(d_mdf["hikayeler"]).split()).value_counts()[-3:]
d_mdf["hikayeler"].apply(lambda x: " ".join(x for x in x.split() if x not in sil))
#################
#Toketnization
#################
nltk.download("punkt")
# !pip install textblob
import textblob
from textblob import TextBlob
TextBlob(d_mdf["hikayeler"][1]).words
d_mdf["hikayeler"].apply(lambda x: TextBlob(x).words)

#################
#Stemming
#################

from nltk.stem import PorterStemmer
st = PorterStemmer()

d_mdf["hikayeler"].apply(lambda x: " ".join(st.stem(i) for i in x.split()))

#################
#Lemmatization
#################
from textblob import Word
nltk.download("wordnet")

d_mdf["hikayeler"].apply(lambda x: " ".join(Word(i).lemmatize() for i in x.split()))

####################
# Matematiksel İşlemler ve Basit Özellik Çıkarımı
####################
#################
#Harf/Karakter sayısı
################
d_mdf["hikayeler"]
o_df = d_mdf.copy()
o_df["hikayeler"].str.len()
o_df["harfsay"]=o_df["hikayeler"].str.len()

#################
#Kelime sayısı
################
o_df["KelimeSay"] =o_df["hikayeler"].apply(lambda x: len(str(x).split(" ")))
#################
#Özel Karakterleri Yakalamak & Saydırmak
################
o_df["ozel_kar_say"] =o_df["hikayeler"].apply(lambda x: len([x for x in x.split() if x.startswith("adventure")]))

#################
#Sayıları Yakalamak & Saydırmak
################
o_df['sayı_sayısı'] = mdf["hikayeler"].apply(lambda x: len([x for x in x.split() if x.isdigit()]))