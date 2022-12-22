from textblob import TextBlob
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics
from sklearn.feature_extraction.text import  TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

#pip install keras
#pip install tensorflow

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

import pandas as pd

data= pd.read_csv("C:/Users/izzet/PycharmProjects/DSMLBC8/NLP/train.tsv", sep="\t")



df.head()

data["Sentiment"].replace(0, value= "negatif", inplace=True)
data["Sentiment"].replace(1, value= "negatif", inplace=True)

data["Sentiment"].replace(3, value= "pozitif", inplace=True)
data["Sentiment"].replace(4, value= "pozitif", inplace=True)

data = data[(data.Sentiment == "negatif") | (data.Sentiment == "pozitif")]

data.groupby("Sentiment").count()

df = pd.DataFrame()
df["text"] =  data["Phrase"]
df["label"] =  data["Sentiment"]
df.head()
##################
#Metin Ön İşleme
###################
######################################
######################################
######################################

#### Noktalama İşaretlerinin Silinmesi
df['text'] =df['text'].str.replace("[^\w\s]","")

######Sayıların Silinmesi
df['text'] = df['text'].str.replace("\d","")
df['text'] = df['text'].apply(lambda x : " ".join(x.lower() for x in x.split()))
#Stopwords
import nltk
from nltk.corpus import stopwords
sw = stopwords.words("english")
df["text"] = df["text"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
#################
#Az  geçen kelimelerin siiinmesi
#################
pd.Series(" ".join(df["text"]).split()).value_counts()
sil = pd.Series(" ".join(df["text"]).split()).value_counts()[-1000:]
df["text"].apply(lambda x: " ".join(x for x in x.split() if x not in sil))
#################
#Lemmatization
#################
from textblob import Word
nltk.download("wordnet")
df["text"].apply(lambda x: " ".join(Word(i).lemmatize() for i in x.split()))

######################################
######################################
######################################
######################################
#DEĞİŞKEN MÜHENDİSLİĞİ
##COUNT VECTORS
##TF_IDF VECTORS
##WORD EMBEDDINGS
##TF = (Bir T teriminin bir dokümanda gözlenme frekansı ) /(dokümandaki toplam terim sayısı)
##IDF = log_e(Toplam doküman sayısı/ içinde t terimi olan belge sayısı)
######################################
######################################
###Test -Train

train_x, test_x, train_y, test_y= model_selection.train_test_split(df["text"], df["label"])

encoder = preprocessing.LabelEncoder()

train_y = encoder.fit_transform(train_y)
test_y = encoder.fit_transform(test_y)

############################
#Count Vectors
############################
vectorizer = CountVectorizer()
vectorizer.fit(train_x)

x_train_count = vectorizer.transform(train_x)
x_test_count = vectorizer.transform(test_x)

vectorizer.get_feature_names()[0:5]

x_train_count.toarray()

############################
#TF_IDF
###########################
tf_idf_word_vectorizer = TfidfVectorizer()
tf_idf_word_vectorizer.fit(train_x)

x_train_tf_idf_word = tf_idf_word_vectorizer.transform(train_x)
x_test_tf_idf_word = tf_idf_word_vectorizer.transform(test_x)

tf_idf_word_vectorizer.get_feature_names()[0:5]

##########################
#n-gram level tf_idf
##########################
tf_idf_ngram_vectorizer = TfidfVectorizer(ngram_range=(2,3))
tf_idf_ngram_vectorizer.fit(train_x)

x_train_tf_idf_ngram = tf_idf_ngram_vectorizer.transform(train_x)
x_test_tf_idf_ngram = tf_idf_ngram_vectorizer.transform(test_x)

##########################
#character level tf_idf
##########################

tf_idf_ngram_chars_vectorizer = TfidfVectorizer(analyzer="char",ngram_range=(2,3))
tf_idf_ngram_chars_vectorizer.fit(train_x)

x_train_tf_idf_chars = tf_idf_ngram_chars_vectorizer.transform(train_x)
x_test_tf_idf_chars = tf_idf_ngram_chars_vectorizer.transform(test_x)

############################################
############################################
############################################
#Makine Öğrenmesi ile Sentiment Sınıflandırması
############################################
#Lojistik Reg

loj = linear_model.LogisticRegression()
loj_model = loj.fit(x_train_count, train_y)
accuracy = model_selection.cross_val_score(loj_model, x_test_count, test_y, cv=10).mean()

print("Count Vectors Doğruluk Oranı :", accuracy)
############################################
loj = linear_model.LogisticRegression()
loj_model = loj.fit(x_train_tf_idf_word, train_y)
accuracy = model_selection.cross_val_score(loj_model, x_test_count, test_y, cv=10).mean()

print("Word-Level TF-IDF Doğruluk Oranı :", accuracy)
############################################
loj = linear_model.LogisticRegression()
loj_model = loj.fit(x_train_tf_idf_ngram, train_y)
accuracy = model_selection.cross_val_score(loj_model, x_test_count, test_y, cv=10).mean()

print("N-GRAM TF-IDF Doğruluk Oranı :", accuracy)
############################################
loj = linear_model.LogisticRegression()
loj_model = loj.fit(x_train_tf_idf_chars, train_y)
accuracy = model_selection.cross_val_score(loj_model, x_test_count, test_y, cv=10).mean()

print("CHARLEVEL Doğruluk Oranı :", accuracy)
############################################
############################################
#Naive Bayes
nb = naive_bayes.MultinomialNB()
nb_model = nb.fit(x_train_count, train_y)
accuracy = model_selection.cross_val_score(nb_model, x_test_count, test_y, cv=10).mean()

print("Count Vectors Doğruluk Oranı :", accuracy)
############################################
nb = naive_bayes.MultinomialNB()
nb_model = nb.fit(x_train_tf_idf_word, train_y)
accuracy = model_selection.cross_val_score(nb_model, x_test_count, test_y, cv=10).mean()

print("Word-Level TF-IDF Doğruluk Oranı :", accuracy)
############################################
nb = naive_bayes.MultinomialNB()
nb_model = nb.fit(x_train_tf_idf_ngram, train_y)
accuracy = model_selection.cross_val_score(loj_model, x_test_count, test_y, cv=10).mean()

print("N-GRAM TF-IDF Doğruluk Oranı :", accuracy)
############################################
nb = naive_bayes.MultinomialNB()
nb_model = nb.fit(x_train_tf_idf_chars, train_y)
accuracy = model_selection.cross_val_score(nb_model, x_test_count, test_y, cv=10).mean()

print("CHARLEVEL Doğruluk Oranı :", accuracy)
############################################
############################################
loj_model.predict("yes i like this film")

yeni_yorum = pd.Series("this film is very nice and good i like it")
v= CountVectorizer()
v.fit(train_x)
yeni_yorum = v.transform(yeni_yorum)
loj_model.predict(yeni_yorum)