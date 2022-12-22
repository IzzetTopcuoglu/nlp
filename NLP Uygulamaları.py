###########################
#NLP Uygulamalar
###########################
#N-GRAM
###########################
import nltk

a="""Bu örneği anlaşılabilmesi için daha uzun bir metin üzerinden göstereceğim. N-Gramlar birlikte kulanılan kelimelerin kombinasyonlarını gösterir"""

import textblob
from textblob import TextBlob
TextBlob(a).ngrams(3)

# [WordList(['Bu', 'örneği', 'anlaşılabilmesi']),
#  WordList(['örneği', 'anlaşılabilmesi', 'için']),
#  WordList(['anlaşılabilmesi', 'için', 'daha']),
#  WordList(['için', 'daha', 'uzun']),
#  WordList(['daha', 'uzun', 'bir']),
#  WordList(['uzun', 'bir', 'metin']),
#  WordList(['bir', 'metin', 'üzerinden']),
#  WordList(['metin', 'üzerinden', 'göstereceğim']),
#  WordList(['üzerinden', 'göstereceğim', 'N-Gramlar']),
#  WordList(['göstereceğim', 'N-Gramlar', 'birlikte']),
#  WordList(['N-Gramlar', 'birlikte', 'kulanılan']),
#  WordList(['birlikte', 'kulanılan', 'kelimelerin']),
#  WordList(['kulanılan', 'kelimelerin', 'kombinasyonlarını']),
#  WordList(['kelimelerin', 'kombinasyonlarını', 'gösterir'])]

###########################
#Part Of Speech tagging (pos)
###########################
nltk.download("averaged_perceptron_tagger")
TextBlob(d_mdf["hikayeler"][2]).tags
# [('redheaded', 'VBN'), ('league', 'NN')]
d_mdf["hikayeler"].apply(lambda x:TextBlob(x).tags)

###########################
#Chunking (shallow parsing)
###########################
pos = d_mdf["hikayeler"].apply(lambda x:TextBlob(x).tags)
cumle = "R and Python are useful data science tools for the new or old data scientists who eager to do efficient data science task"

pos = TextBlob(cumle).tags
pos

reg_exp="NP: {<DT>?<JJ>*<NN>}"
rp= nltk.RegexpParser(reg_exp)
sonuclar = rp.parse(pos)
sonuclar.draw()

###########################
#Named Entity Recognition
###########################
from nltk import word_tokenize, pos_tag, ne_chunk
nltk.download("maxent_ne_chunker")
nltk.download("words")
cumle = "Hadley is a creative person who works for R studio AND he attended the conference in Newyork last year"
print(ne_chunk(pos_tag(word_tokenize(cumle))))